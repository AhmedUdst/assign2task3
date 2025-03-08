import streamlit as st
import faiss
import numpy as np
from mistralai import Mistral, UserMessage
from bs4 import BeautifulSoup
import requests
import sys

# Ensure Python version compatibility
REQUIRED_PYTHON_VERSION = (3, 7)
MAX_PYTHON_VERSION = (3, 10)
if not (REQUIRED_PYTHON_VERSION <= sys.version_info[:2] <= MAX_PYTHON_VERSION):
    raise RuntimeError("This script requires Python >=3.7 and <3.11")

# API Key Setup
API_KEY = "xjCgy80GBjYF4qDbKke2ZI98Q8jxoinY"

def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=API_KEY)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

def mistral_chat(user_message):
    client = Mistral(api_key=API_KEY)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Load Policy Data
def load_policy_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error loading policy: {str(e)}"

# Build FAISS Index
def build_faiss_index(text_chunks):
    text_embeddings = get_text_embedding(text_chunks)
    embeddings = np.array([emb.embedding for emb in text_embeddings])
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, text_chunks

# Retrieve Relevant Chunks
def retrieve_relevant_chunks(query, index, text_chunks, top_k=2):
    query_embedding = np.array([get_text_embedding([query])[0].embedding])
    D, I = index.search(query_embedding, k=top_k)
    return [text_chunks[i] for i in I.tolist()[0] if i < len(text_chunks)]

# Intent Classification
def classify_intent(query, policy_descriptions):
    prompt = f"""
    Given the following policies:
    {policy_descriptions}
    Determine which policy is most relevant to the following question: {query}
    Respond with only the policy name.
    """
    return mistral_chat(prompt).strip()

# Streamlit UI
st.title("UDST Policy Chatbot - Agentic RAG")

# Policy URLs (Expanded to 20 policies)
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
    # Add more policies up to 20
}

# Load Policies on Demand
if "policy_data" not in st.session_state:
    st.session_state["policy_data"] = {}
    for policy, url in policy_urls.items():
        st.session_state["policy_data"][policy] = load_policy_data(url)

# Intent Classification and Query Processing
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    policy_descriptions = "\n".join([f"- {p}" for p in policy_urls.keys()])
    selected_policy = classify_intent(query, policy_descriptions)
    
    if selected_policy in policy_urls:
        policy_text = st.session_state["policy_data"].get(selected_policy, "Error: Policy data not available.")
        if "Error" in policy_text:
            st.error(policy_text)
        else:
            chunks = [policy_text[i:i+512] for i in range(0, len(policy_text), 512)]
            index, chunks = build_faiss_index(chunks)
            retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
            
            if retrieved_chunks:
                prompt = f"""
                Context information:
                {retrieved_chunks}
                Given the context, answer the query: {query}
                """
                response = mistral_chat(prompt)
                st.text_area("Answer:", response, height=200)
            else:
                st.error("No relevant information found in the selected policy.")
    else:
        st.error("No relevant policy found.")

# Ensure Streamlit runs properly with correct dependencies
st.write("Ensure you have the correct Python version and all dependencies installed. If issues persist, try updating pip using: \"pip install --upgrade pip\" and reinstall dependencies with \"pip install -r requirements.txt\"")
