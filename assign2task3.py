import streamlit as st
import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from dotenv import load_dotenv

# Load environment variables
os.environ["MISTRAL_API_KEY"] = "WxuATixGO6kp5LQ2ilW1jLRiD5IFibV8"
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

# Initialize session state for relevant policies
if "relevant_policies" not in st.session_state:
    st.session_state.relevant_policies = []
    st.session_state.relevant_documents = None

st.write("Enter your queries (first question: get relevant policies, subsequent questions: ask about them):")
user_input = st.text_area("Enter your prompt:", height=200)

if user_input:
    inputs = user_input.split("\n")
    responses = []
    
    for i, query in enumerate(inputs):
        query = query.strip()
        if not query:
            continue
        
        if i == 0:
            # Step 1: Find relevant policies
            policy_query_lower = query.lower()
            relevant_policies = [name for name in policy_texts.keys() if any(word in policy_query_lower for word in name.lower().split())]
            st.session_state.relevant_policies = relevant_policies
            
            if relevant_policies:
                policy_list = "\n".join([f"- {policy_name}: {policy_urls[policy_name]}" for policy_name in relevant_policies])
                responses.append(f"**Relevant Policies:**\n{policy_list}")
                
                # Create a new index with only relevant documents
                relevant_documents = [Document(text=policy_texts[name], metadata={"name": name}) for name in relevant_policies]
                st.session_state.relevant_documents = VectorStoreIndex.from_documents(relevant_documents).as_query_engine()
            else:
                responses.append("No matching policies found.")
        else:
            # Step 2: Answer specific policy-related question using previously retrieved policies
            if st.session_state.relevant_policies and st.session_state.relevant_documents:
                response = st.session_state.relevant_documents.query(query)
                responses.append(f"**Response:** {response.response}")
            else:
                responses.append("No relevant policies found to answer this question.")
    
    for response in responses:
        st.write(response)
