import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import tempfile  # For temporary file handling
import PyPDF2  # For handling PDF uploads
from langchain.schema import Document

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Google Gemini API for text generation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
output_parser = StrOutputParser()

# Streamlit setup
st.title('RAG Demo with Google Gemini')
input_text = st.text_input("Search the topic you want")

# Step 1: PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extract text from the uploaded PDF using PyPDF2
    reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ''
    for page_num in range(len(reader.pages)):
        extracted_text += reader.pages[page_num].extract_text()

    st.write("Extracted text from PDF:")
    st.write(extracted_text)

    # Convert extracted text to a document format LangChain can process
    documents = [Document(page_content=extracted_text)]  # Simple dictionary with text content

    # Step 2: Generate embeddings using Google Vertex AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Ensure this is the correct model
    vectorstore = FAISS.from_documents(documents, embeddings)  # Store embeddings in FAISS for similarity search

    # Step 3: Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever()  # Use the vectorstore to retrieve relevant document chunks

    # Step 4: Initialize RetrievalQA
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Or "map_reduce", depending on your use case
        retriever=retriever,
        return_source_documents=True  # Optional, if you want source documents in the output
    )

    # Step 5: Process user query
    if input_text:
        st.write("Processing your query...")
        response = chain({"query": input_text})  # Use a dictionary for input
        st.write("Response:")
        st.write(response['result'])  # Access the response text
        
