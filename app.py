import os
import shutil
import pickle
import numpy as np
import faiss
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer

# Assumed custom module from your snippet
from bookmarks import get_firefox_bookmarks

# ---------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")
EMBED_FILE = os.path.join(DATA_DIR, "embeddings.npy")
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------
# 2. PERSISTENCE LAYER (IO Handling)
# ---------------------------------------------------------
def save_vector_db(index, titles, urls, embeddings):
    """Responsible only for writing data to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    
    with open(META_FILE, "wb") as f:
        pickle.dump({"titles": titles, "urls": urls}, f)
    
    np.save(EMBED_FILE, embeddings)

def load_vector_db():
    """
    Responsible only for reading data from disk.
    Returns None if the database is incomplete or missing.
    """
    required_files = [INDEX_FILE, META_FILE, EMBED_FILE]
    if not all(os.path.exists(p) for p in required_files):
        return None

    try:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        embeddings = np.load(EMBED_FILE)
        return index, meta["titles"], meta["urls"], embeddings
    except Exception as e:
        # If data is corrupted, return None so we can rebuild
        return None

def clear_data_cache():
    """Responsible for cleaning up persistent storage."""
    shutil.rmtree(DATA_DIR, ignore_errors=True)

# ---------------------------------------------------------
# 3. BUSINESS LOGIC LAYER (Core Intelligence)
# ---------------------------------------------------------
def create_index_from_bookmarks(model):
    """
    Responsible for fetching data, generating embeddings, 
    and building the FAISS index.
    """
    bookmarks = get_firefox_bookmarks()
    if not bookmarks:
        raise ValueError("No bookmarks found to index.")

    titles = [bm["title"] for bm in bookmarks]
    urls = [bm["url"] for bm in bookmarks]
    
    # Generate embeddings
    embeddings = model.encode(titles, convert_to_numpy=True)
    
    # Build Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, titles, urls, embeddings

# ---------------------------------------------------------
# 4. ORCHESTRATION LAYER (The Controller)
# ---------------------------------------------------------
@st.cache_resource
def get_search_engine():
    """
    Orchestrates the loading or creation of the search engine.
    This is the only entry point the UI needs to know about.
    """
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Try to load existing DB
    db_data = load_vector_db()
    
    if db_data is not None:
        index, titles, urls, embeddings = db_data
        return index, model, titles, urls, embeddings

    # 2. If load failed, create new DB
    index, titles, urls, embeddings = create_index_from_bookmarks(model)
    
    # 3. Save for next time
    save_vector_db(index, titles, urls, embeddings)
    
    return index, model, titles, urls, embeddings

# ---------------------------------------------------------
# 5. PRESENTATION LAYER (UI Components)
# ---------------------------------------------------------
def render_wordcloud(titles, header="Bookmark Title Cloud"):
    """Visual component for WordCloud."""
    st.subheader(f"📘 {header}")
    text = " ".join(titles)
    # Basic validation
    if not text.strip():
        st.info("Not enough text to generate cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def main():
    st.title("🔍 Firefox Bookmark Search")

    # Initialize Session State
    if "results_shown" not in st.session_state:
        st.session_state.results_shown = 10

    # Initialize Backend
    with st.spinner("Initializing vector database..."):
        try:
            index, model, titles, urls, embeddings = get_search_engine()
            st.success(f"Ready! {len(titles)} bookmarks loaded.")
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

    # Settings Section
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("### 📈 Visualization & Search Settings")
        col1, col2 = st.columns(2)
        with col1:
            max_titles = len(titles) if titles else 200
            top_n = st.slider("Bookmarks to visualize", 10, max_titles, min(30, max_titles), step=5)
        with col2:
            results_slider = st.slider("🔢 Results to show", 5, 50, st.session_state.results_shown, 5)
            st.session_state.results_shown = results_slider

    # Search Input
    query = st.text_input("Search your bookmarks")

    # Search Logic
    if query:
        # Fetch slightly more to allow for "Load More" functionality
        fetch_limit = st.session_state.results_shown + 10
        query_vec = model.encode([query], convert_to_numpy=True)
        
        # FAISS search
        distances, indices = index.search(query_vec, fetch_limit)
        
        st.subheader("Top Matches:")
        
        # Determine valid count (handle cases where index returns -1 for empty slots)
        valid_indices = [i for i in indices[0] if i != -1 and i < len(titles)]
        count_to_show = min(len(valid_indices), st.session_state.results_shown)

        for i in range(count_to_show):
            idx = valid_indices[i]
            dist = distances[0][i]
            st.markdown(f"""
            **{titles[idx]}** [{urls[idx]}]({urls[idx]})  
            *Distance: {dist:.2f}*
            """)

        # Action Buttons
        c1, c2 = st.columns([1, 1])
        with c1:
            if count_to_show < len(valid_indices):
                if st.button("🔽 Load More"):
                    st.session_state.results_shown += 10
                    st.rerun()
        with c2:
            if st.button("📊 Visualize Search Results"):
                matched_titles = [titles[i] for i in valid_indices[:count_to_show]]
                render_wordcloud(matched_titles, "Search Results")

    # Global Visualizations
    if st.button("🌐 Visualize All Bookmarks"):
        render_wordcloud(titles[:top_n], "All Bookmarks")

    # Rebuild Database Action
    if st.button("🔁 Rebuild Database"):
        clear_data_cache()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()