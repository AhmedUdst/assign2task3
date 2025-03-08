import streamlit as st
import requests
from llama_index.core import SimpleWebPageReader
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Streamlit UI
st.title("UDS Policies Query Assistant")
st.write("Search and retrieve relevant UDS policies or ask any question.")

# User Input
query = st.text_input("Enter your query:")

# Fetch UDS policies from web pages
uds_policy_urls = [
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy", 
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy"# Adjust with actual policy URLs
]

st.write("Fetching UDS policies...")
documents = SimpleWebPageReader(html_to_text=True).load_data(uds_policy_urls)
st.success("UDS policies loaded successfully!")

# Initialize LLM & Embedding
api_key = "LsTDsPmjahnJz2Xlie33gaGnAOKx1IM6"
llm = MistralAI(api_key=api_key)
Settings.llm = llm
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)

# Indexing UDS policies
nodes = [doc.get_text() for doc in documents]
vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine()

# Query Execution
if query:
    response = query_engine.query(query)
    st.subheader("Response:")
    st.write(str(response))
    
# Deployment Notes
st.sidebar.header("Deployment Instructions")
st.sidebar.markdown("1. Run `streamlit run app.py` to start locally.")
st.sidebar.markdown("2. Deploy on GitHub by including `requirements.txt` and using Streamlit Sharing.")
st.sidebar.markdown("3. API Key is hardcoded for now but should be securely managed in deployment.")
