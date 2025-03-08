import streamlit as st
import os
import nest_asyncio
from llama_index.core import VectorStoreIndex, Settings
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

# Define policy URLs
policy_urls = {
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy", 
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy", 
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy", 
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy", 
    "Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy", 
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy", 
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy", 
}

# Set LLM globally
Settings.llm = MistralAI(api_key=MISTRAL_API_KEY)

# Set Embedding Model to Mistral
Settings.embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)

st.write("Enter a prompt to get the relevant policy name:")
first_prompt = st.text_input("Enter your first prompt:")

if first_prompt:
    # Simulate query engine response with a best-matching policy
    policy_name = max(policy_urls.keys(), key=lambda name: name.lower() in first_prompt.lower())
    st.write("Relevant Policy Name:", policy_name)
    st.write("Policy URL:", policy_urls[policy_name])
    
    second_prompt = st.text_input("Enter your second prompt for more details:")
    
    if second_prompt:
        st.write("For more details, visit the policy URL above.")
