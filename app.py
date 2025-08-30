from transformers import pipeline
import pyttsx3
import PyPDF2
import requests
import streamlit as st
import tempfile
import os

# ----------------------------
# 1. Load Model
# ----------------------------
generator = pipeline(
    "text-generation",
    model="gpt2",        # Small & lightweight, good for testing
    device_map="cpu",     
    torch_dtype="auto"
)

# ----------------------------
# 2. AI Text Analysis
# ----------------------------
def analyze_text(text):
    result = generator(
        text, 
        max_length=300,   # adjust output length
        do_sample=True, 
        temperature=0.7
    )
    return result[0]['generated_text']

# ----------------------------
# 3. Text-to-Speech (TTS) with Streamlit support
# ----------------------------
def speak(text):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        filename = fp.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# ----------------------------
# 4. Extract text from PDF
# ----------------------------
def extract_pdf_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ----------------------------
# 5. Extract text from a URL
# ----------------------------
def extract_link_text(url):
    response = requests.get(url)
    return response.text

# ----------------------------
# 6. Streamlit UI
# ----------------------------
st.title("📘 Smart Learning Assistant (EchoVerse)")
st.write("Upload PDF / Enter URL / Paste Text")

choice = st.radio("Choose Input Type", ["PDF", "Link", "Text"])
text = ""

if choice == "PDF":
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file:
        text = extract_pdf_text(pdf_file)
        st.text_area("Extracted Text", text[:1000])  # preview first 1000 chars

elif choice == "Link":
    url = st.text_input("Enter URL")
    if st.button("Fetch"):
        text = extract_link_text(url)
        st.text_area("Extracted Text", text[:1000])

elif choice == "Text":
    text = st.text_area("Enter Text Here")

# Use session_state to persist explanation
if "explanation" not in st.session_state:
    st.session_state.explanation = ""

if st.button("Explain"):
    if text.strip():
        st.session_state.explanation = analyze_text(text)

if st.session_state.explanation:
    st.text_area("Explanation", st.session_state.explanation, height=200)
    if st.button("Speak"):
        audio_file = speak(st.session_state.explanation)
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        os.remove(audio_file)  # Clean up temp file