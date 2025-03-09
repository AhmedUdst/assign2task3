import streamlit as st
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import os

# Set API Key
os.environ["MISTRAL_API_KEY"] = "WxuATixGO6kp5LQ2ilW1jLRiD5IFibV8"
api_key = os.getenv("MISTRAL_API_KEY")

# Load Data
documents = SimpleDirectoryReader(input_files=["documents.txt"]).load_data()

# Split Data
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

# Define LLM and Embedding Model
llm = MistralAI(api_key=api_key)
embedding_model = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)

# Create Indexes
summary_index = SummaryIndex(nodes[:10])
vector_index = VectorStoreIndex(nodes[:10])

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

# Streamlit UI
st.title("UDST Policy Information System")
st.write("Enter a prompt to get the most relevant UDST policies or answers to your questions.")

# User Input
user_prompt = st.text_area("Enter your prompt:")

if st.button("Submit"):
    if user_prompt:
        response = query_engine.query(user_prompt)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt.")
