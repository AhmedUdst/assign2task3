import streamlit as st
import os
import faiss
import numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralEmbeddingModel
from bs4 import BeautifulSoup
import requests

# Set API Key
os.environ["MISTRAL_API_KEY"] = "xjCgy80GBjYF4qDbKke2ZI98Q8jxoinY"
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize LLM and Embeddings
llm = MistralAI(api_key=api_key)
embed_model = MistralEmbedding(api_key=api_key)

# Function to Load Policy Data
def load_policy_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error loading policy: {str(e)}"

# Define Policy URLs
policy_urls = {
    "Student Conduct Policy": "https://www.udst.edu.qa/.../student-conduct-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/.../academic-schedule-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/.../student-attendance-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/.../student-appeals-policy",
    "Graduation Policy": "https://www.udst.edu.qa/.../graduation-policy",
    "Academic Standing Policy": "https://www.udst.edu.qa/.../academic-standing-policy",
    "Transfer Policy": "https://www.udst.edu.qa/.../transfer-policy",
    "Admissions Policy": "https://www.udst.edu.qa/.../admissions-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/.../final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/.../registration-policy",
}

# Load Policies on Demand
policy_data = {}
for policy, url in policy_urls.items():
    policy_data[policy] = load_policy_data(url)

# Split Policies into Chunks
splitter = SentenceSplitter(chunk_size=512)
policy_chunks = {}
for policy, text in policy_data.items():
    policy_chunks[policy] = splitter.get_nodes_from_documents([text])

# Build FAISS Index
def build_faiss_index(text_chunks):
    embeddings = np.array([embed_model.get_text_embedding(chunk.get_text()) for chunk in text_chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks

policy_indexes = {}
for policy, chunks in policy_chunks.items():
    policy_indexes[policy] = build_faiss_index(chunks)

# Retrieve Relevant Chunks
def retrieve_relevant_chunks(query, index, text_chunks, top_k=2):
    query_embedding = np.array([embed_model.get_text_embedding(query)])
    D, I = index.search(query_embedding, k=top_k)
    return [text_chunks[i].get_text() for i in I.tolist()[0] if i < len(text_chunks)]

# Intent Classification
def classify_intent(query):
    policy_list = "\n".join([f"- {p}" for p in policy_urls.keys()])
    prompt = f"""
    Given the following policies:
    {policy_list}
    Determine which policy is most relevant to the following question: {query}
    Respond with only the policy name.
    """
    return llm.complete(prompt).strip()

# Streamlit UI
st.title("UDST Policy Chatbot - Agentic RAG")
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    selected_policy = classify_intent(query)
    
    if selected_policy in policy_indexes:
        index, chunks = policy_indexes[selected_policy]
        retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
        
        if retrieved_chunks:
            prompt = f"""
            Context information:
            {retrieved_chunks}
            Given the context, answer the query: {query}
            """
            response = llm.complete(prompt)
            st.text_area("Answer:", response, height=200)
        else:
            st.error("No relevant information found in the selected policy.")
    else:
        st.error("No relevant policy found.")
