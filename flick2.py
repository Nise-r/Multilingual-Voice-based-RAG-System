import tkinter as tk
import threading
import pyaudio
import wave
import io
from TTS.api import TTS
import sounddevice as sd
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from piper import PiperVoice
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

#faster-whisper model
model = WhisperModel("base", compute_type="int8")

#initializing llm
os.environ['GROQ_API_KEY'] ='YOUR_API_KEY'
llm = init_chat_model('groq:llama-3.3-70b-versatile')

#prompt for respnse generation
generalPrompt = ChatPromptTemplate.from_messages([
    ("system","""You are an execellent chatbot,you talk in human-like manner and generate answers from reference data.
    Answer the user question based on the reference data.
    Make sure the answer is clear and complete."""),
    ("human","Reference Data: {reference}"+"\n\n "+"Query: {input}")
])

#tts voices 
englishPipe = PiperVoice.load(model_path="./voices/en_GB-alan-medium.onnx")
spanishPipe = PiperVoice.load(model_path="./voices/es_ES-davefx-medium.onnx")
chinesePipe = PiperVoice.load(model_path="./voices/zh_CN-huayan-medium.onnx")

#Embedding used for vector-database
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={"device": "cpu"})

#initializing chroma db
vector_store = Chroma(
    collection_name="finalCollenction",
    embedding_function=embedding,
    persist_directory="./finalChroma"
)

#recording parameters for audio
chunk = 1024  
sample_format = pyaudio.paInt16
channels = 1
fs = 16000  
# filename = "output.wav"

#global variables
is_recording = False
frames = []
audio_np = None
blink_state = True
sttext = ""
response = ""
language = ""
is_transcribing = False


def startRecord():
    global is_recording,frames
    is_recording = True
    frames = []
    blink_label(recording_label,color="red",txt="● Recording")
    
    def record():
        global audio_np
        p = pyaudio.PyAudio()  
        stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)
        
        print("Recording...")
        while is_recording:
                data = stream.read(chunk)
                frames.append(data)
                
        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        print('Finished recording')
    threading.Thread(target=record).start()

def stt():
    global sttext,is_transcribing,language

#     segments, _ = model.transcribe(audio_np, language="en")
    segments, info = model.transcribe(audio_np, task="translate")
    language = info.language
    print("-"*10)
    print("Detected Language: ",language)
    print("-"*10)
    
    sttext = ""
    
    for segment in segments:
        sttext+=segment.text
        
    if language == 'zh':
        transcribed = GoogleTranslator(source='auto', target='zh-CN').translate(sttext)
    else:
        transcribed = GoogleTranslator(source='auto', target=language).translate(sttext)
    
    print("Transcribed: ",transcribed)
    print("-"*10)
    
    is_transcribing = False
    transcribing_label.place_forget()
    
    transcribed_text.delete("1.0",tk.END)
    transcribed_text.insert("1.0",transcribed)
    
    threading.Thread(target=rag).start()
    
def speechToText():
    global sttext, is_transcribing
    is_transcribing = True

    blink_label(transcribing_label,color="blue",txt="...Transcribing")

    threading.Thread(target=stt).start()
    
    
def toggle_recording():
    global is_recording
    
    if not is_recording:
        startRecord()
        record_button.config(text="Stop Recording")

    else:
        is_recording = False
        record_button.config(text="Start Recording")
        recording_label.place_forget()
       
        root.after(1000,speechToText)

        
        
def blink_label(widget, color="red",txt = "● Recording"):
    def toggle():
        nonlocal visible
        if is_recording or is_transcribing:
            visible = not visible

            widget.config(text=txt if visible else "")
            root.after(500, toggle)
        else:
              widget.config(text="")
    visible = True
    toggle()

def ttspeech():
    global response
    
    buffer = io.BytesIO()
    
    # Open a wave writer over that buffer
    with wave.open(buffer, 'wb') as wav_file:
        if language=='en':
            englishPipe.synthesize(response, wav_file)
        elif language=='es':
            spanishPipe.synthesize(response, wav_file)
        elif language=='zh':
            chinesePipe.synthesize(response, wav_file)
        else:
            print("ERROR LANGUAGE NOT SUPPORTED")
            return 

    buffer.seek(0)


    with wave.open(buffer, 'rb') as wav_reader:
        framerate = wav_reader.getframerate()
        frames = wav_reader.readframes(wav_reader.getnframes())

    audio_array = np.frombuffer(frames, dtype=np.int16)

    sd.play(audio_array, samplerate=framerate)
    sd.wait()
    
    
def rag():
    global sttext, response
    
    docs = vector_store.similarity_search(sttext,k=3)
    response = docs[0].page_content
    
    chain = generalPrompt |llm
    result = chain.invoke({
        "input":sttext,
        "reference":response
    })

    if language=='zh':
        response = GoogleTranslator(source='auto', target='zh-CN').translate(result.content)
    else:
        response = GoogleTranslator(source='auto', target=language).translate(result.content)
    
    response_text.delete("1.0",tk.END)
    response_text.insert("1.0",response)
    
    threading.Thread(target=ttspeech).start()


root = tk.Tk()
root.geometry("800x600")
root.title("Voice Recorder")

        
record_button = tk.Button(root, text="Start Recording", command=toggle_recording)
record_button.pack(pady=(20,10))

recording_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), fg="red")
recording_label.pack()

transcribing_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), fg="blue")
transcribing_label.pack()

transcribed_label = tk.Label(root,text="Transcribed:", font=("Helvetica", 14, "bold"), fg="black")
transcribed_label.pack(pady=(10,0))

transcribed_text = tk.Text(root,width=90,height=2,font=("Helvetica", 14, "bold"),fg="black")
transcribed_text.pack(pady=(0,40))


response_label = tk.Label(root,text="Response:", font=("Helvetica", 14, "bold"), fg="black")
response_label.pack(pady=0)
response_text = tk.Text(root,width=90,height=8,font=("Helvetica", 14, "bold"),fg="black")
response_text.pack(pady=0)

exit = tk.Button(root, text="Quit", command=root.destroy)
exit.pack(pady=40)



root.mainloop()