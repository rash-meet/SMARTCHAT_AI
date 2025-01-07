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
from gtts import gTTS
import speech_recognition as sr  # For speech-to-text

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
    # Using the new StuffDocumentsChain
    #chain = StuffDocumentsChain(llm=model, prompt=prompt)
    return chain

# Generate TTS audio from text
def generate_tts(text, filename="response.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Process user input
def user_input(user_question, tts_enabled, use_gemini, pdf_docs):
    try:
        if use_gemini:
            # Use Gemini for response generation (may include knowledge beyond the PDFs)
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            response = model.invoke(user_question)
            # Adjust based on the response structure
            response_text = response.content  # Assuming the 'content' attribute holds the answer
        else:
            # Use PDF-based QA
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)

            chain = get_conversational_chain()
            response = chain.invoke({"input_documents": docs, "question": user_question})
            response_text = response["output_text"]

        st.write("Reply: ", response_text)

        if tts_enabled:
            # Generate TTS audio
            audio_file = generate_tts(response_text)
            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Speech-to-Text Function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
    return ""  # Return empty string if speech is not recognized

def main():
    st.set_page_config(page_title="SMARTCHAT")
    st.header("SMARTCHAT: AI INTERACTION MODEL BY RS😊")

    tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=True)
    use_gemini = st.sidebar.checkbox("Enable Gemini AI", value=False)  # New toggle for Gemini AI
    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("No text found in the uploaded PDFs.")
        else:
            st.error("Please upload at least one PDF file.")

    # User input for question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Add button for speech-to-text
    if st.button("Speak"):
        speech_text = recognize_speech()  # Call the speech-to-text function
        if speech_text:
            user_question = speech_text  # Set recognized text to question input

    # Check if there is a user question
    if user_question:
        user_input(user_question, tts_enabled, use_gemini, pdf_docs)

if __name__ == "__main__":
    main()
