import streamlit as st
from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
import speech_recognition
from langchain import LLMChain, PromptTemplate
import os
from colorama import Fore
import pyttsx3 as tts
import webbrowser



st.title("Optimus 🔧")

def history(txt):
    with st.expander("Brain history "):
        st.write(txt)

def bot_reponse(txt):
    with st.expander("Bot repsonse 🔧"):
        st.write(txt)

speaker = tts.init()
voice = speaker.getProperty('voices')
speaker.setProperty('voice',
                    'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\'
                    'Speech\\Voices\\Tokens'
                    '\\MSTTS_V110_frFR_PaulM')
speaker.setProperty("rate", 175)
#pour la voix du chatbot
def dire(text):
    speaker.say(text)
    speaker.runAndWait()
#pour le microphone
def ver_microphone():
    recognizer = speech_recognition.Recognizer()
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            text = recognizer.listen(mic, phrase_time_limit=15)
            text = recognizer.recognize_google(audio, language='fr-FR')
            text = text.lower()
            return text
    except speech_recognition.UnknownValueError():
        recognizer = speech_recognition.Recognizer()
        pritn("error")

#Pour lancer le model d'inteligence articielle 
#set the voice of the bot
load_dotenv()
model_name = os.environ.get('MODEL_3') 
model_n_ctx = os.environ.get('MODEL_N_CTX') # Convert to integer
temperature = os.environ.get("TEMPERATURE") 
#for the title of the ap
#initialisation of the model
template = """tu es un assitant intelligent qui s'appelle optimus.
{chat_history}
Human: {human_input}
Chatbot:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = LlamaCpp(
        model_path=model_name,
        #n_ctx = 16000,
        n_batch=100,
        n_gpu_layers=2,
        temperature=0.1,
        max_tokens=2048,
        n_ctx=16000,
        threads=3,
        #temperature= 0.2,
        verbose = True,
        streaming=True
        #threads=2
)
Fore.LIGHTMAGENTA_EX
llm_chain = LLMChain(
    llm=llm,
    verbose=True,
    memory=memory,
    prompt=prompt,
)
x = 0
myquestion = st.chat_input("Your request")
while True:
    if myquestion == None:
        st.write('write a question')
    else:
        txt = llm_chain.run(human_input=myquestion)
        history(memory)
        bot_reponse(txt)

