import os
import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from bookmarks import get_firefox_bookmarks
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import shutil

DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")

def save_vector_db(index, titles, urls):
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"titles": titles, "urls": urls}, f)

def load_vector_db():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta["titles"], meta["urls"]

@st.cache_resource
def init_index():
    index, titles, urls = load_vector_db()
    if index:
        return index, SentenceTransformer("all-MiniLM-L6-v2"), titles, urls

    bookmarks = get_firefox_bookmarks()
    titles = [bm["title"] for bm in bookmarks]
    urls = [bm["url"] for bm in bookmarks]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(titles, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    save_vector_db(index, titles, urls)
    return index, model, titles, urls

def visualize_titles(model, titles, top_n=30, neighbors_k=3):
    st.subheader("🔗 Title Relationship Graph")
    embeddings = model.encode(titles[:top_n], convert_to_numpy=True)
    nn = NearestNeighbors(n_neighbors=neighbors_k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    G = nx.Graph()
    for i, title in enumerate(titles[:top_n]):
        G.add_node(i, label=title)
        for j in indices[i][1:]:  # skip self
            G.add_edge(i, j)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, {i: titles[i][:25] + "..." for i in G.nodes}, font_size=9)
    plt.axis("off")
    st.pyplot(plt)

# -----------------------------
# Streamlit App Starts Here
# -----------------------------

st.title("🔍 Firefox Bookmark Search (Semantic)")

with st.spinner("Loading or building vector database..."):
    try:
        index, model, titles, urls = init_index()
        st.success(f"Ready! {len(titles)} bookmarks loaded.")
    except Exception as e:
        st.error(f"Failed to initialize vector DB: {e}")
        st.stop()

# -----------------------------
# Visualization Controls
# -----------------------------
st.markdown("### 📈 Visualization Settings")
col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Bookmarks to visualize", 10, min(len(titles), 200), 30, step=5)
with col2:
    neighbors_k = st.slider("Related nodes per title", 1, 10, 3)

# -----------------------------
# Search Input
# -----------------------------
query = st.text_input("Search your bookmarks")

if "results_shown" not in st.session_state:
    st.session_state.results_shown = 10

# -----------------------------
# Search Logic
# -----------------------------
if query:
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, 50)  # Up to 50 results

    st.subheader("Top Matches:")
    count_to_show = min(len(I[0]), st.session_state.results_shown)

    for i in range(count_to_show):
        idx = I[0][i]
        st.markdown(f"**{titles[idx]}**  \n[{urls[idx]}]({urls[idx]})  \n_(Distance: {D[0][i]:.2f})_")

    col1, col2 = st.columns([1, 1])
    with col1:
        if count_to_show < len(I[0]):
            if st.button("🔽 Load More"):
                st.session_state.results_shown += 10
                st.rerun()
    with col2:
        if st.button("📊 Visualize Search Results"):
            matched_titles = [titles[I[0][i]] for i in range(count_to_show)]
            visualize_titles(model, matched_titles, top_n=min(top_n, len(matched_titles)), neighbors_k=neighbors_k)

# -----------------------------
# Visualize All Bookmarks
# -----------------------------
if st.button("🌐 Visualize All Bookmarks"):
    visualize_titles(model, titles, top_n=top_n, neighbors_k=neighbors_k)

# -----------------------------
# Rebuild Button
# -----------------------------
if st.button("🔁 Rebuild Database"):
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()
