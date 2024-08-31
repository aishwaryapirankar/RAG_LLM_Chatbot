# Import necessary libraries
import os
import time
import streamlit as st
from streamlit_float import *
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader # Previously, from langchain_community.document_loaders import WebBaseLoader
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

# Load the environmental variables 
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY'] # Get the Groq API key

# Advanced RAG setup
if "vector" not in st.session_state:
    st.session_state.value = "Processing..."
    st.session_state.embeddings = HuggingFaceBgeEmbeddings() # Previously, st.session_state.loader = WebBaseLoader("https://www.industryacademiacommunity.com/faqs")
    st.session_state.loader=PyPDFLoader("FAQs.pdf")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Model initialization
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "mixtral-8x7b-32768") 

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
st.markdown(html_temp,unsafe_allow_html=True)

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
                        print("Response time :",time.process_time()-start)

                    if response:
                        with st.chat_message("assistant"):
                            answer = response['answer']
                            st.markdown(answer)
                            speak_answer(answer)  
                            

# Accept user's text input
if prompt := st.chat_input("Hi and Welcome! Please ask your query regarding the Industry Academia Community Program here"):
    start=time.process_time()
    response=retrieval_chain.invoke({"input": prompt})
    print("Response time :",time.process_time()-start)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        response = st.write(response['answer'])