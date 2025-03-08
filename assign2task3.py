import streamlit as st
import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Apply nest_asyncio (for running async tasks in notebooks if needed)
nest_asyncio.apply()

# Streamlit UI
st.title("Agentic RAG with Mistral AI")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_path = f"./{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully!")
    
    # Process document
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    
    llm = MistralAI(api_key=MISTRAL_API_KEY)
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    
    st.write("Document indexed successfully. Enter a query below:")
    query = st.text_input("Ask a question about the document:")
    
    if query:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        st.write("Response:", response)
