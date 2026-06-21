#  AI Meeting Intelligence Assistant

An end-to-end AI system that converts raw meeting audio into structured insights, including transcripts, summaries, and intelligent question-answering using a Retrieval-Augmented Generation (RAG) pipeline.

It automates meeting understanding by combining speech recognition, large language models, and semantic search.

---

##  Features

*  **Audio Transcription** using Whisper (faster-whisper)
*  **AI Meeting Summarization** using Gemini LLM
*  Extracts:
  ** Executive Summary
  ** Action Items
  ** Key Decisions
  ** Risks & Concerns
  ** Open Questions
*  **RAG-based Q&A System** over meeting transcripts
*  Semantic search using FAISS vector database
*  Interactive Streamlit web interface
*  End-to-end automated pipeline from audio → insights

---

##  Architecture

```
Audio File
   ↓
Preprocessing (mono + normalization)
   ↓
Whisper Transcription
   ↓
Gemini Summarization
   ↓
FAISS Vector Store (Embeddings)
   ↓
RAG-based Question Answering
   ↓
Streamlit UI Output
```

---

##  Tech Stack

* Python
* Streamlit
* faster-whisper
* Google Gemini API
* LangChain
* FAISS
* Sentence Transformers
* NumPy
* SoundFile

---

##  Project Structure

```
ai-meeting-assistant/
│
├── app.py
├── modules/
│   ├── transcription.py
│   ├── summary.py
│   ├── rag.py
│   ├── preprocessing.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  How It Works

1. Upload a meeting recording
2. Audio is preprocessed (mono + normalized)
3. Whisper model generates transcript
4. Gemini LLM converts transcript into structured insights
5. FAISS creates embeddings for semantic search
6. Users can ask questions about the meeting via RAG

---

## Live Demo
🔗 Live demo:https://k9p3cbjw6o6tt2btqvbahx.streamlit.app/

##  Installation (Local Setup)

```
git clone https://github.com/your-username/ai-meeting-assistant.git
cd ai-meeting-assistant

pip install -r requirements.txt

streamlit run app.py
```

---

## Environment Variables

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key"
HF_TOKEN = "your_huggingface_token"
```

---

