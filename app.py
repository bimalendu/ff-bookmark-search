import os
import shutil
import pickle
import numpy as np
import faiss
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Set, Optional

# Assumed custom module
# from bookmarks import get_firefox_bookmarks 
# Mocking the import for standalone functionality if the file is missing
try:
    from bookmarks import get_firefox_bookmarks
except ImportError:
    def get_firefox_bookmarks():
        # Mock data for demonstration if module is missing
        return [
            {"title": "Streamlit Documentation", "url": "https://docs.streamlit.io"},
            {"title": "Python SOLID Principles", "url": "https://realpython.com/solid-principles-python/"},
            {"title": "FAISS Indexing Tutorial", "url": "https://github.com/facebookresearch/faiss"},
            {"title": "Sentence Transformers", "url": "https://www.sbert.net/"},
        ] * 5

# =========================================================
# 1. CONFIGURATION (Single Source of Truth)
# =========================================================
class AppConfig:
    DATA_DIR = "data"
    INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
    META_FILE = os.path.join(DATA_DIR, "meta.pkl")
    EMBED_FILE = os.path.join(DATA_DIR, "embeddings.npy")
    IGNORED_FILE = os.path.join(DATA_DIR, "ignored.pkl")
    MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_RESULTS_LIMIT = 10
    DEFAULT_IGNORED_LIMIT = 5

    @staticmethod
    def ensure_data_dir():
        os.makedirs(AppConfig.DATA_DIR, exist_ok=True)

    @staticmethod
    def clear_cache():
        shutil.rmtree(AppConfig.DATA_DIR, ignore_errors=True)

# =========================================================
# 2. PERSISTENCE LAYER (SRP: Handling Disk I/O only)
# =========================================================
class PersistenceManager:
    """Handles saving and loading of index, metadata, and ignored lists."""
    
    @staticmethod
    def save_vector_db(index: faiss.Index, titles: List[str], urls: List[str], embeddings: np.ndarray):
        AppConfig.ensure_data_dir()
        faiss.write_index(index, AppConfig.INDEX_FILE)
        with open(AppConfig.META_FILE, "wb") as f:
            pickle.dump({"titles": titles, "urls": urls}, f)
        np.save(AppConfig.EMBED_FILE, embeddings)

    @staticmethod
    def load_vector_db() -> Optional[Tuple[faiss.Index, List[str], List[str], np.ndarray]]:
        required = [AppConfig.INDEX_FILE, AppConfig.META_FILE, AppConfig.EMBED_FILE]
        if not all(os.path.exists(p) for p in required):
            return None
        try:
            index = faiss.read_index(AppConfig.INDEX_FILE)
            with open(AppConfig.META_FILE, "rb") as f:
                meta = pickle.load(f)
            embeddings = np.load(AppConfig.EMBED_FILE)
            return index, meta["titles"], meta["urls"], embeddings
        except Exception:
            return None

    @staticmethod
    def save_ignored(ignored_set: Set[str]):
        AppConfig.ensure_data_dir()
        with open(AppConfig.IGNORED_FILE, "wb") as f:
            pickle.dump(ignored_set, f)

    @staticmethod
    def load_ignored() -> Set[str]:
        if not os.path.exists(AppConfig.IGNORED_FILE):
            return set()
        try:
            with open(AppConfig.IGNORED_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return set()

# =========================================================
# 3. CORE LOGIC (SRP: Search, Indexing, and Filtering)
# =========================================================
class SearchEngine:
    """Encapsulates embedding generation, indexing, and search logic."""
    
    def __init__(self):
        self.model = SentenceTransformer(AppConfig.MODEL_NAME)
        self.index = None
        self.titles = []
        self.urls = []
        self.embeddings = None
        self._initialize_db()

    def _initialize_db(self):
        """Loads DB from disk or creates a new one (Lazy Initialization)."""
        data = PersistenceManager.load_vector_db()
        if data:
            self.index, self.titles, self.urls, self.embeddings = data
        else:
            self.rebuild_index()

    def rebuild_index(self):
        """Fetches bookmarks and builds the FAISS index."""
        bookmarks = get_firefox_bookmarks()
        if not bookmarks:
            raise ValueError("No bookmarks found.")
        
        self.titles = [bm["title"] for bm in bookmarks]
        self.urls = [bm["url"] for bm in bookmarks]
        self.embeddings = self.model.encode(self.titles, convert_to_numpy=True)
        
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        PersistenceManager.save_vector_db(self.index, self.titles, self.urls, self.embeddings)

    def search(self, query: str, ignored_urls: Set[str]) -> List[Tuple[int, float]]:
        """
        Returns filtered results.
        Returns: List of (index, distance) tuples.
        """
        if not self.index:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        # Search all to ensure we can filter effectively
        distances, indices = self.index.search(query_vec, len(self.titles))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.titles):
                url = self.urls[idx]
                if url not in ignored_urls:
                    results.append((idx, distances[0][i]))
        return results

    def get_bookmark(self, idx: int) -> Tuple[str, str]:
        return self.titles[idx], self.urls[idx]

    def get_title_map(self) -> Dict[str, str]:
        """Returns a dict mapping URL -> Title."""
        return dict(zip(self.urls, self.titles))

# =========================================================
# 4. PRESENTATION LAYER (UI Rendering)
# =========================================================
class UIManager:
    """Handles all Streamlit rendering and state management."""
    
    def __init__(self, engine: SearchEngine):
        self.engine = engine
        self._init_session_state()

    def _init_session_state(self):
        if "results_limit" not in st.session_state:
            st.session_state.results_limit = AppConfig.DEFAULT_RESULTS_LIMIT
        if "ignored_limit" not in st.session_state:
            st.session_state.ignored_limit = AppConfig.DEFAULT_IGNORED_LIMIT
        if "ignored_urls" not in st.session_state:
            st.session_state.ignored_urls = PersistenceManager.load_ignored()

    def _render_wordcloud(self, titles: List[str], header: str):
        st.subheader(f"📘 {header}")
        text = " ".join(titles)
        if not text.strip():
            st.info("Not enough text to generate cloud.")
            return
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    def _handle_toggle(self, url: str):
        """Callback for ignore toggle."""
        ignored = st.session_state.ignored_urls
        if url in ignored:
            ignored.remove(url)
        else:
            ignored.add(url)
        PersistenceManager.save_ignored(ignored)

    def render_sidebar(self):
        with st.sidebar:
            st.header("Actions")
            if st.button("🔁 Rebuild Database"):
                AppConfig.clear_cache()
                st.cache_resource.clear()
                st.session_state.clear()
                st.rerun()
            
            st.markdown("---")
            st.info(f"Bookmarks Loaded: {len(self.engine.titles)}")
            st.info(f"Ignored Items: {len(st.session_state.ignored_urls)}")

    def render_search_results(self, query: str):
        # Fetch results filtered by ignore list
        results = self.engine.search(query, st.session_state.ignored_urls)
        total_results = len(results)
        
        # Pagination
        visible_results = results[:st.session_state.results_limit]

        st.subheader(f"Matches ({total_results})")
        
        for idx, dist in visible_results:
            title, url = self.engine.get_bookmark(idx)
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(f"{url} | Score: {dist:.2f}")
            with c2:
                is_ignored = url in st.session_state.ignored_urls
                st.toggle("Ignore", value=is_ignored, key=f"tog_{idx}", 
                          on_change=self._handle_toggle, args=(url,))
            st.divider()

        # Load More Button
        if st.session_state.results_limit < total_results:
            if st.button("🔽 Load More Results"):
                st.session_state.results_limit += 10
                st.rerun()

        # Visualization Button
        if visible_results and st.button("📊 Visualize These Results"):
            titles = [self.engine.titles[i] for i, _ in visible_results]
            self._render_wordcloud(titles, "Search Context")

    def render_ignored_list(self):
        ignored = list(st.session_state.ignored_urls)
        if not ignored:
            return

        st.markdown("---")
        with st.expander(f"🚫 Ignored Items ({len(ignored)})"):
            visible_ignored = ignored[:st.session_state.ignored_limit]
            title_map = self.engine.get_title_map()

            for url in visible_ignored:
                title = title_map.get(url, "Unknown Title")
                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    st.write(f"**{title}**")
                    st.caption(url)
                with c2:
                    st.toggle("Ignore", value=True, key=f"ign_{abs(hash(url))}",
                              on_change=self._handle_toggle, args=(url,))
                st.divider()

            if st.session_state.ignored_limit < len(ignored):
                if st.button("🔽 Load More Ignored"):
                    st.session_state.ignored_limit += 5
                    st.rerun()

    def render_main(self):
        st.title("🔍 Bookmark Smart Search")
        self.render_sidebar()
        
        query = st.text_input("Search bookmarks...")
        if query:
            self.render_search_results(query)
        
        self.render_ignored_list()

# =========================================================
# 5. ORCHESTRATION (Main Application Entry)
# =========================================================
@st.cache_resource
def get_engine():
    """Singleton-like pattern via Streamlit cache."""
    return SearchEngine()

def main():
    try:
        engine = get_engine()
        ui = UIManager(engine)
        ui.render_main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        if st.button("Reset Application"):
            AppConfig.clear_cache()
            st.rerun()

if __name__ == "__main__":
    main()