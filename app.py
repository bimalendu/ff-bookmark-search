import os
import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from bookmarks import get_firefox_bookmarks

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

st.title("🔍 Firefox Bookmark Search (Semantic)")

with st.spinner("Loading or building vector database..."):
    try:
        index, model, titles, urls = init_index()
        st.success(f"Ready! {len(titles)} bookmarks loaded.")
    except Exception as e:
        st.error(f"Failed to initialize vector DB: {e}")
        st.stop()

query = st.text_input("Search your bookmarks")

if query:
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, 5)

    st.subheader("Top Matches:")
    for i, idx in enumerate(I[0]):
        st.markdown(f"**{titles[idx]}**  \n[{urls[idx]}]({urls[idx]})  \n_(Distance: {D[0][i]:.2f})_")

if st.button("🔁 Rebuild Database"):
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    st.cache_resource.clear()
    st.rerun()
