# 🗣️ Multilingual-Voice-based-RAG-System 🎙️

This project is an interactive voice-based chatbot that takes spoken input from the user, transcribes it using Speech-to-Text (STT), retrieves relevant information from a vector database (RAG), generates a response using an LLM, and finally speaks the response back to the user using Text-to-Speech (TTS).

## 🎨 Images



## 🚀 Features

- **Speech-to-Text (STT)**: Transcribes user speech using faster-whisper.
- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant reference data from a ChromaDB using HuggingFace embeddings.
- **LLM Response Generation**:  Uses Groq's LLaMA 3.3 70B model to generate human-like answers based on reference data.
- **Fast Text-to-Speech (TTS)**: Generates audio responses in English, Spanish,Chinese and many more using Piper.
- **Language Detection & Translation**: Supports multiple languages and auto-translation using Google Translator.
- **Tkinter UI**:Simple interface with recording, transcribing, and response display.

---

## 🛠️ Tech Stack

| Component                | Library/Tool                             |
|--------------------------|------------------------------------------|
| Speech-to-Text (STT)     | [faster-whisper](https://github.com/guillaumekln/faster-whisper)   |
| Vector Database (RAG)    | [ChromaDB](https://www.trychroma.com/) + HuggingFace multilingual Embeddings    |
| LLM                      | [Groq LLaMA 3.3 70B](https://groq.com/)                             |
| Text-to-Speech (TTS)     | [Piper](https://github.com/rhasspy/piper)                            |
| Translation              | [GoogleTranslator (deep-translator)](https://github.com/nidhaloff/deep-translator) |
| Audio Processing         | PyAudio, SoundDevice, Numpy            |
| GUI                      | Tkinter                                 |

---

## 📦 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nise-r/Multilingual-Voice-based-RAG-System.git
cd Multilingual-Voice-based-RAG-System
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download/Prepare Required Models

faster-whisper model (base in this project). It will be automatically downloaded on first use.

### 4️⃣ Set Environment Variables
update directly in the code (as seen in the script):
```bash
os.environ['GROQ_API_KEY'] = 'YOUR_API_KEY'
```
### 5️⃣ Run the Application
```bash
python main.py
```

## 🎛️ Usage Instructions
1. Launch the application. A window with buttons and text fields will appear.
2. Click Start Recording to capture your voice input.
3. The system will:
    - Transcribe the input (STT)
    - Retrieve relevant data (RAG)
    - Generate a response (LLM)
    - Speak the response (TTS)
4. View the transcription and response in the UI.
