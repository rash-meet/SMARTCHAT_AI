import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydub import AudioSegment
import speech_recognition as sr
import tempfile

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please check your environment variables.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''  # Handle cases where extraction fails
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Get the conversational chain using StuffDocumentsChain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context, just say, "Answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Speech-to-Text Function
def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Error with Google Speech Recognition API: {e}")
    return ""

# Main application
def main():
    st.set_page_config(page_title="SMARTCHAT")
    st.header("SMARTCHAT: AI INTERACTION MODEL BY RS")

    tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=True)
    use_gemini = st.sidebar.checkbox("Enable Gemini AI", value=False)
    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. Ready for questions!")
                else:
                    st.error("No text found in the uploaded PDFs.")
        else:
            st.error("Please upload at least one PDF file.")

    # User input for question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Audio file upload for speech-to-text
    audio_file = st.file_uploader("Upload an audio file for speech recognition", type=["wav", "mp3"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_file.read())
            temp_path = temp_file.name

        # Convert MP3 to WAV if necessary
        if audio_file.name.endswith(".mp3"):
            sound = AudioSegment.from_mp3(temp_path)
            temp_path = temp_path.replace(".mp3", ".wav")
            sound.export(temp_path, format="wav")

        # Recognize speech
        recognized_text = recognize_speech(temp_path)
        if recognized_text:
            user_question = recognized_text

    if user_question:
        st.write(f"You asked: {user_question}")
        # Use Gemini or FAISS-based retrieval
        try:
            if use_gemini:
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                response = model.invoke(user_question)
                response_text = response.content
            else:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = vector_store.similarity_search(user_question)

                chain = get_conversational_chain()
                response = chain.invoke({"input_documents": docs, "question": user_question})
                response_text = response["output_text"]

            st.write("Reply: ", response_text)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
