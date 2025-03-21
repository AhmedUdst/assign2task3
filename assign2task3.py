import streamlit as st
import pickle
import faiss
import numpy as np
import time
import math

from mistralai import Mistral, UserMessage

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Enhanced Agentic RAG with Policy Selection",
    layout="centered"  # Single page, no sidebar
)

# -------------------------------
# Load Data: index + rag_data
# -------------------------------
with open("rag_data.pkl", "rb") as f:
    data = pickle.load(f)

valid_chunks = data["chunks"]
valid_sources = data["sources"]
api_key = data["api_key"]

# If we stored chunk_embeddings, retrieve them. Otherwise, we'll re-embed on the fly
chunk_embeddings = data.get("chunk_embeddings", None)

# Load FAISS index
index = faiss.read_index("rag_index.faiss")
embeddings_dim = index.d



# -------------------------------
# Helper Functions
# -------------------------------
def get_text_embedding(text_list, batch_size=20):
    """
    Uses Mistral to get embeddings for each text in text_list.
    Returns a list of embedding vectors.
    """
    client = Mistral(api_key=api_key)
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        try:
            response = client.embeddings.create(model="mistral-embed", inputs=batch)
            all_embeddings.extend(r.embedding for r in response.data)
            time.sleep(2)  # small sleep to reduce rate-limit risk
        except Exception as e:
            st.error(f"Error retrieving embeddings: {e}")
            for _ in batch:
                all_embeddings.append(None)
    return all_embeddings


def agentic_policy_selection(question, policy_titles):
    """
    Uses Mistral to pick the top 2 most relevant policies to the user’s question.
    """
    client = Mistral(api_key=api_key)
    prompt = (
        f"Given the following policies: {', '.join(policy_titles)}. "
        f"Which two policies are most relevant to answer the question: '{question}'? "
        "Please provide your answer as a comma-separated list with no additional text."
    )
    messages = [UserMessage(content=prompt)]
    
    try:
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        answer = chat_response.choices[0].message.content.strip()
        selected = [p.strip() for p in answer.split(",")]
        return selected[:2]
    except Exception as e:
        st.error(f"Error selecting policies: {e}")
        return []


def rag_query(question, k=10):
    """
    1) Agentic policy selection
    2) Filter chunks to those from the selected policies
    3) Search with FAISS among the filtered set
    4) Construct final prompt with top-k chunks
    5) Generate answer with Mistral
    Returns (answer, selected_policies).
    """
    # All distinct policy titles from our sources
    all_policy_titles = list(
        {src.split(" - ")[0].replace("Policy: ", "") for src in valid_sources}
    )

    # 1) Agentic step
    selected_policies = agentic_policy_selection(question, all_policy_titles)
    if not selected_policies:
        selected_policies = all_policy_titles  # fallback if none

    # 2) Filter to chunks that contain any of the selected policy names
    filtered_indices = [
        i for i, src in enumerate(valid_sources)
        if any(sp in src for sp in selected_policies)
    ]
    if len(filtered_indices) == 0:
        # fallback to everything if no match
        filtered_indices = list(range(len(valid_chunks)))

    # 3) Build a sub-index (FAISS) with just the relevant embeddings
    sub_index = faiss.IndexFlatL2(embeddings_dim)

    # We either use chunk_embeddings from pickle or re-embed
    global chunk_embeddings  # so we can save them if they're newly created
    if chunk_embeddings is None:
        st.warning("No saved embeddings found. Re-embedding all chunks. Might be slow.")
        chunk_embeddings = get_text_embedding(valid_chunks)

        # Save them to the data dict and rewrite rag_data.pkl
        data["chunk_embeddings"] = chunk_embeddings
        with open("rag_data.pkl", "wb") as f:
            pickle.dump(data, f)

    # Filter to relevant subset
    filtered_embeddings = [chunk_embeddings[i] for i in filtered_indices]
    filtered_embeddings_np = np.array(filtered_embeddings, dtype="float32")

    sub_index.add(filtered_embeddings_np)

    # 4) Query embedding
    q_emb = get_text_embedding([question])
    if not q_emb or q_emb[0] is None:
        return "Error generating embeddings for your query.", selected_policies
    q_emb_np = np.array([q_emb[0]], dtype="float32")

    # 5) Search in the sub-index
    top_k = min(k, len(filtered_embeddings_np))
    D, I = sub_index.search(q_emb_np, top_k)

    # Retrieve top-k chunks
    retrieved_chunks = [valid_chunks[filtered_indices[idx]] for idx in I[0]]
    retrieved_srcs   = [valid_sources[filtered_indices[idx]] for idx in I[0]]

    # Build the context
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_text += f"Chunk {i+1}:\n{chunk}\n{retrieved_srcs[i]}\n---\n"

    final_prompt = f"""
You are given the following context information. Use it to answer the user's question accurately.
If the information needed is not in the context, say "I don't have enough information to answer this question."

Context information:
---------------------
{context_text}
---------------------

Question: {question}

Please provide a comprehensive answer based solely on the provided context.
Include references to the policy used to get that answer formatted like this: Policy: Name - (Policy URL). Don't mention the chunk numbers.

"""

    # 6) Generate the final answer
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=final_prompt)]
    try:
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        final_answer = chat_response.choices[0].message.content
    except Exception as e:
        final_answer = f"Error generating response: {e}"

    return final_answer, selected_policies

# -------------------------------
# UI Layout
# -------------------------------
st.write("Enter your queries (first question: get relevant policies, subsequent questions: ask about them):")

# --- Chat box: user enters question here ---

main_container = st.container()
with main_container:
    # st.markdown("## 💬 Chat Input")
    user_query = st.chat_input("Your question about these policies:")

    if user_query:
        if user_query.strip():
            with st.spinner("Thinking.."):
                final_answer, selected = rag_query(user_query.strip())
            
            # Show which policies the agentic step selected
            st.markdown("**Policies the system found most relevant:**")
            if selected:
                for s in selected:
                    st.write(f"- {s}")
            else:
                st.write("No specific policy found.")
            
            st.markdown("### Answer")
            st.write(final_answer)
        else:
            st.warning("Please enter a question first.")


