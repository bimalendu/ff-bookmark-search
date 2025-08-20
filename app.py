import os
import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from bookmarks import get_firefox_bookmarks
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shutil

DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")
EMBED_FILE = os.path.join(DATA_DIR, "embeddings.npy")

def save_vector_db(index, titles, urls, embeddings):
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"titles": titles, "urls": urls}, f)
    np.save(EMBED_FILE, embeddings)

def load_vector_db():
    if not all(os.path.exists(p) for p in [INDEX_FILE, META_FILE, EMBED_FILE]):
        return None, None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    embeddings = np.load(EMBED_FILE)
    return index, meta["titles"], meta["urls"], embeddings

@st.cache_resource
def init_index():
    index, titles, urls, embeddings = load_vector_db()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # FIXED CONDITION HERE:
    if None not in (index, titles, urls, embeddings):
        return index, model, titles, urls, embeddings

    bookmarks = get_firefox_bookmarks()
    titles = [bm["title"] for bm in bookmarks]
    urls = [bm["url"] for bm in bookmarks]
    embeddings = model.encode(titles, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    save_vector_db(index, titles, urls, embeddings)
    return index, model, titles, urls, embeddings

def title_wordcloud(titles, header="Bookmark Title Cloud"):
    st.subheader(f"📘 {header}")
    text = " ".join(titles)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# -----------------------------
# Streamlit App Starts Here
# -----------------------------
st.title("🔍 Firefox Bookmark Search")

# Initialize session state
if "results_shown" not in st.session_state:
    st.session_state.results_shown = 10  # Default number of results

with st.spinner("Initializing vector database..."):
    try:
        index, model, titles, urls, embeddings = init_index()
        st.success(f"Ready! {len(titles)} bookmarks loaded.")
    except Exception as e:
        st.error(f"Failed to initialize vector DB: {e}")
        st.stop()

# -----------------------------
# Toggle Settings Section
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    st.markdown("### 📈 Visualization & Search Settings")
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Bookmarks to visualize", 10, min(len(titles), 200), 30, step=5)
    with col2:
        results_slider_value = st.slider(
            "🔢 Results to show",
            min_value=5,
            max_value=50,
            value=st.session_state.results_shown,
            step=5,
        )
        if results_slider_value != st.session_state.results_shown:
            st.session_state.results_shown = results_slider_value

# -----------------------------
# Search Input
# -----------------------------
query = st.text_input("Search your bookmarks")

# -----------------------------
# Search Logic
# -----------------------------
if query:
    # FIXED: Prefetch more results than shown to enable Load More button
    results_to_fetch = st.session_state.results_shown + 10

    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, results_to_fetch)

    st.subheader("Top Matches:")
    count_to_show = min(len(I[0]), st.session_state.results_shown)

    for i in range(count_to_show):
        idx = I[0][i]
        st.markdown(f"""
        **{titles[idx]}**  
        [{urls[idx]}]({urls[idx]})  
        Distance: {D[0][i]:.2f}  
        """)

    col1, col2 = st.columns([1, 1])
    with col1:
        if count_to_show < len(I[0]):
            if st.button("🔽 Load More"):
                st.session_state.results_shown += 10
                st.rerun()
    with col2:
        if st.button("📊 Visualize Search Results"):
            matched_titles = [titles[I[0][i]] for i in range(count_to_show)]
            title_wordcloud(matched_titles, "Search Results")

# -----------------------------
# Visualize All Bookmarks
# -----------------------------
if st.button("🌐 Visualize All Bookmarks"):
    title_wordcloud(titles[:top_n], "All Bookmarks")

# -----------------------------
# Rebuild Button
# -----------------------------
if st.button("🔁 Rebuild Database"):
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()
