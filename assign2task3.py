import streamlit as st
import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from dotenv import load_dotenv

# Load environment variables
os.environ["MISTRAL_API_KEY"] = "LsTDsPmjahnJz2Xlie33gaGnAOKx1IM6"
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Apply nest_asyncio (for running async tasks in notebooks if needed)
nest_asyncio.apply()

# Streamlit UI
st.title("Agentic RAG with Mistral AI")

# Define policy URLs and content (for indexing purposes)
policy_texts = {
    "Student Conduct Policy": "This policy governs student behavior and disciplinary actions...",
    "Academic Schedule Policy": "This policy outlines the academic calendar and scheduling rules...",
    "Student Attendance Policy": "This policy explains attendance requirements and consequences of absenteeism...",
    "Student Appeals Policy": "This policy details the process for students to appeal decisions...",
    "Graduation Policy": "This policy describes graduation requirements and processes...",
    "Academic Standing Policy": "This policy defines academic performance standards...",
    "Transfer Policy": "This policy outlines the transfer of credits and student mobility...",
    "Admissions Policy": "This policy sets the criteria and procedures for student admissions...",
    "Final Grade Policy": "This policy explains the grading system and final grade calculations...",
    "Registration Policy": "This policy provides guidelines for course registration and enrollment...",
}

policy_urls = {
    name: f"https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/{name.lower().replace(' ', '-')}"
    for name in policy_texts.keys()
}

# Set LLM globally
Settings.llm = MistralAI(api_key=MISTRAL_API_KEY)
Settings.embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)

# Create documents for indexing
documents = [Document(text=policy_texts[name], metadata={"name": name}) for name in policy_texts]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.write("Enter a prompt to get relevant policies:")
first_prompt = st.text_input("Enter your first prompt:")

if first_prompt:
    first_prompt_lower = first_prompt.lower()
    relevant_policies = [name for name in policy_texts.keys() if any(word in first_prompt_lower for word in name.lower().split())]
    
    if relevant_policies:
        st.write("Relevant Policies:")
        for policy_name in relevant_policies:
            st.write(f"- {policy_name}: {policy_urls[policy_name]}")
    else:
        st.write("No matching policies found.")
    
    second_prompt = st.text_input("Enter your second prompt for more details:")
    
    if second_prompt:
        response = query_engine.query(second_prompt)
        st.write("Response:", response.response)
