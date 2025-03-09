import streamlit as st
import time
import random
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import os

# Set API Key
os.environ["MISTRAL_API_KEY"] = "e72ywIR9EFLA0Din6EZbQqjhAGMAZCqX"
api_key = os.getenv("MISTRAL_API_KEY")

# Load Data
documents = SimpleDirectoryReader(input_files=["documents.txt"]).load_data()

# Split Data
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents, show_progress=False)  # Process all policies from documents.txt  # Process only first 10 policies to reduce API load

# Ensure All 20 Policies Are Included
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes, insert_batch_size=10)  # Increase batch size for efficiency  # Limit batch size to reduce API strain

# Define LLM and Embedding Model
llm = MistralAI(api_key=api_key)
embedding_model = MistralAIEmbedding(model_name="mistral-embed-v1", api_key=api_key)

# Ensure Mistral is the default embedding model
from llama_index.core import Settings
Settings.embed_model = embedding_model

# Define Query Engines
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
vector_query_engine = vector_index.as_query_engine()

# Define Query Engine Tools
summary_tool = QueryEngineTool.from_defaults(query_engine=summary_query_engine, description="Summarization of UDST policies.")
vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine, description="Retrieve specific UDST policy information.")

# Define Router Query Engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True
)

# Retry Logic for API Rate Limits
def retry_query(query_engine, user_prompt, retries=3, wait_time=5):
    import logging
    # logging disabled to optimize performance
    for attempt in range(retries):
        try:
            response = query_engine.query(user_prompt)
            return response
        except Exception as e:
            logging.error(f"Mistral API Error: {str(e)}")
            st.warning(f"API Rate Limit Exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time + random.uniform(0, 2))  # Add jitter
    return "API limit exceeded. Please try again later."

# Streamlit UI
st.title("UDST Policy Information System")
st.write("Enter a prompt to get the most relevant UDST policies or answers to your questions.")

# User Input
user_prompt = st.text_input("Enter your prompt:")

if user_prompt.strip():  # Avoid empty or accidental whitespace queries
        response = retry_query(query_engine, user_prompt)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")
