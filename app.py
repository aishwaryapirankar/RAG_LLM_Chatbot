# Import necessary libraries
import os
import time
import streamlit as st
from streamlit_float import *
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Import Speech Synthesis packages
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import gtts
import requests
from twilio.rest import Client
from dotenv import load_dotenv
from flask import Flask, request

# Load the environmental variables 
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']  # Get the Groq API key
twilio_sid = os.environ['TWILIO_ACCOUNT_SID']
twilio_auth_token = os.environ['TWILIO_AUTH_TOKEN']
twilio_phone_number = os.environ['TWILIO_PHONE_NUMBER']
instagram_access_token = os.environ['INSTAGRAM_ACCESS_TOKEN']  # Instagram access token

# Initialize Flask app for webhook handling
app = Flask(__name__)

# Advanced RAG setup
if "vector" not in st.session_state:
    st.session_state.value = "Processing..."
    st.session_state.embeddings = HuggingFaceBgeEmbeddings()
    st.session_state.loader = PyPDFLoader("FAQs.pdf")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Model initialization
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Format context 
prompt = ChatPromptTemplate.from_template(
    """
    Use the following piece of context to answer the question asked.
    Please provide the most accurate response based on the question. 
    Don't mention the provided context in the response.
    If the input is thanks, say welcome as response.
    If the input is a greeting, say hello as response.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Create document and retrieval chain 
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit ChatBot Interface
html_temp = """ <div style="background-image: linear-gradient(to right, #223755, #FEC600, #ED7524); padding:20px">
    <h2 style="color:white;text-align:center;">Cloud Counselage Chatbot</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

# Display chat conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Save chat history 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process speech to text
def listen(mp3_file):    
    recognizer = sr.Recognizer()
    with sr.AudioFile(mp3_file) as source:            
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.write("Could not understand audio")

# Process text to speech
def speak_answer(answer):
    tts = gtts.gTTS(text=answer, lang='en')  # text to speech
    tts.save("answer.mp3") 
    st.audio("answer.mp3", autoplay=True)  
    os.remove("answer.mp3")  

# Accept user's speech input
if prompt := st.container():
    with prompt:
        audio_bytes = audio_recorder()

    if audio_bytes:
        with st.spinner("Transcribing..."):
            mp3_file = "temp_audio.mp3"
            with open(mp3_file, "wb") as f:
                f.write(audio_bytes)

                user_speech = listen(mp3_file)
                if user_speech:
                    with st.chat_message("user"):
                        st.markdown(user_speech)
                        os.remove(mp3_file)

                        start = time.process_time()
                        response = retrieval_chain.invoke({"input": user_speech})
                        print("Response time :", time.process_time() - start)

                    if response:
                        with st.chat_message("assistant"):
                            answer = response['answer']
                            st.markdown(answer)
                            speak_answer(answer)  

# Accept user's text input
if prompt := st.chat_input("Hi and Welcome! Please ask your query regarding the Industry Academia Community Program here"):
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time :", time.process_time() - start)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        st.write(response['answer'])

# Multiple Channel Integration

# Whatsapp
def send_whatsapp_message(to, message):
    client = Client(twilio_sid, twilio_auth_token)
    message = client.messages.create(
        body=message,
        from_=f'whatsapp:{twilio_phone_number}',  # Twilio WhatsApp number
        to=f'whatsapp:{to}'
    )
    return message.sid

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if 'From' in data and 'Body' in data:
        user_message = data['Body']
        phone_number = data['From']
        
        # Process the message with the chatbot
        response = retrieval_chain.invoke({"input": user_message})
        answer = response['answer']
        
        # Send the response back to the user via WhatsApp
        send_whatsapp_message(phone_number, answer)
    return 'OK', 200

# Instagram
def fetch_instagram_messages():
    url = f'https://graph.instagram.com/me/messages?access_token={instagram_access_token}'
    response = requests.get(url)
    return response.json()

def send_instagram_message(user_id, message):
    url = f'https://graph.facebook.com/v13.0/me/messages'
    payload = {
        'recipient': {'id': user_id},
        'message': {'text': message}
    }
    headers = {
        'Authorization': f'Bearer {instagram_access_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# Webhook to receive Instagram messages
@app.route('/instagram_webhook', methods=['POST'])
def instagram_webhook():
    data = request.json
    if 'entry' in data:
        for entry in data['entry']:
            if 'messaging' in entry:
                for messaging_event in entry['messaging']:
                    sender_id = messaging_event['sender']['id']
                    message_text = messaging_event['message']['text']
                    
                    # Process the message with the chatbot
                    response = retrieval_chain.invoke({"input": message_text})
                    answer = response['answer']
                    
                    # Send the response back to the user via Instagram (you'll need to implement sending via Instagram)
                    send_instagram_message(sender_id, answer)
    return 'OK', 200

# SMS
def send_sms(to, message):
    client = Client(twilio_sid, twilio_auth_token)
    message = client.messages.create(
        body=message,
        from_=twilio_phone_number,  # Twilio SMS number
        to=to
    )
    return message.sid

# Webhook to receive SMS messages
@app.route('/sms_webhook', methods=['POST'])
def sms_webhook():
    data = request.form
    if 'From' in data and 'Body' in data:
        user_message = data['Body']
        phone_number = data['From']
        
        # Process the message with the chatbot
        response = retrieval_chain.invoke({"input": user_message})
        answer = response['answer']
        
        # Send the response back to the user via SMS
        send_sms(phone_number, answer)
    return 'OK', 200

if __name__ == '__main__':
    app.run(port=5000)