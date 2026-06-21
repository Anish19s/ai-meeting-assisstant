import streamlit as st
import os
import torchaudio

# ----------------------------
# MUST be FIRST
# ----------------------------
os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")

st.title("AI Meeting Intelligence Assistant")

st.write("APP STARTED ✔️")

# ----------------------------
# Lazy imports (stable)
# ----------------------------
from modules.summary import generate_summary
from modules.rag import generate_vectorstores, q_and_a
from google import genai

from modules.transcription import transcribe
from modules.audio_preprocessing import preproc


# ----------------------------
# Gemini Client
# ----------------------------
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# ----------------------------
# Session State
# ----------------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Meeting Audio",
    type=["wav", "mp3", "m4a", "ogg"]
)

# ----------------------------
# Save File
# ----------------------------
if uploaded_file:

    file_path = "meeting.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Audio uploaded successfully!")

    # ----------------------------
    # Analyze
    # ----------------------------
    if st.button("Analyze Meeting"):

        # ----------------------------
        # PREPROCESSING STEP
        # ----------------------------
        with st.spinner("Preprocessing audio..."):
            waveform, sr = preproc(file_path)

        # ----------------------------
        # TRANSCRIPTION
        # ----------------------------
        with st.spinner("Transcribing audio..."):
            try:
                transcript = transcribe(waveform,sr)
                st.session_state.transcript = transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.session_state.transcript = ""

        # ----------------------------
        # SUMMARY
        # ----------------------------
        if st.session_state.transcript:
            with st.spinner("Generating summary..."):
                summary = generate_summary(
                    st.session_state.transcript,
                    client
                )
                st.session_state.summary = summary

            # ----------------------------
            # VECTOR STORE (RAG)
            # ----------------------------
            with st.spinner("Building knowledge base..."):
                st.session_state.vectorstore = generate_vectorstores(
                    st.session_state.transcript
                )

            st.success("Analysis Complete ✔️")

# ----------------------------
# SUMMARY UI
# ----------------------------
if st.session_state.summary:
    st.subheader("📌 Meeting Summary")
    st.write(st.session_state.summary)

# ----------------------------
# TRANSCRIPT UI
# ----------------------------
if st.session_state.transcript:
    st.subheader("📝 Transcript")

    st.text_area(
        "Full Transcript",
        value=st.session_state.transcript,
        height=300
    )

# ----------------------------
# RAG Q&A
# ----------------------------
if st.session_state.vectorstore:

    st.subheader("💬 Ask Questions")

    query = st.text_input("Ask something about the meeting")

    if query:
        with st.spinner("Searching..."):
            answer = q_and_a(
                st.session_state.vectorstore,
                query,
                client
            )

        st.markdown("### Answer")
        st.write(answer)