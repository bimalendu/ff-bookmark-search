import os
import shutil
import pickle
import sqlite3
import numpy as np
import faiss
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Set, Optional

# Mock import for standalone functionality
try:
    from bookmarks import get_firefox_bookmarks
except ImportError:
    def get_firefox_bookmarks():
        # Fallback mock data
        return [
            {"title": "Streamlit Documentation", "url": "https://docs.streamlit.io"},
            {"title": "Python Performance Optimization", "url": "https://realpython.com/"},
            {"title": "FAISS Indexing", "url": "https://github.com/facebookresearch/faiss"},
            {"title": "SQLite vs Pickle", "url": "https://www.sqlite.org/"},
        ] * 10

# =========================================================
# 1. CONFIGURATION
# =========================================================
class AppConfig:
    DATA_DIR = "data"
    INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
    META_FILE = os.path.join(DATA_DIR, "meta.pkl")
    EMBED_FILE = os.path.join(DATA_DIR, "embeddings.npy")
    DB_FILE = os.path.join(DATA_DIR, "app.db")  # New SQLite DB
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
# 2. PERSISTENCE LAYER (SQLite + FAISS)
# =========================================================
class DatabaseHandler:
    """Handles structured data via SQLite to prevent I/O blocking."""
    
    def __init__(self):
        AppConfig.ensure_data_dir()
        self.conn = sqlite3.connect(AppConfig.DB_FILE, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ignored_urls (
                    url TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_ignored(self, url: str):
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO ignored_urls (url) VALUES (?)", (url,))

    def remove_ignored(self, url: str):
        with self.conn:
            self.conn.execute("DELETE FROM ignored_urls WHERE url = ?", (url,))

    def load_ignored_set(self) -> Set[str]:
        """Reads all ignored URLs into a set for fast O(1) lookups in memory."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT url FROM ignored_urls")
        return {row[0] for row in cursor.fetchall()}

class PersistenceManager:
    """Handles Vector DB files."""
    
    @staticmethod
    def save_vector_data(index: faiss.Index, titles: List[str], urls: List[str], embeddings: np.ndarray):
        AppConfig.ensure_data_dir()
        faiss.write_index(index, AppConfig.INDEX_FILE)
        with open(AppConfig.META_FILE, "wb") as f:
            pickle.dump({"titles": titles, "urls": urls}, f)
        np.save(AppConfig.EMBED_FILE, embeddings)

    @staticmethod
    def load_vector_data() -> Optional[Tuple[faiss.Index, List[str], List[str], np.ndarray]]:
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

# =========================================================
# 3. CORE LOGIC (Optimized Search Engine)
# =========================================================
class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer(AppConfig.MODEL_NAME)
        self.db_handler = DatabaseHandler()
        self.index = None
        self.titles = []
        self.urls = []
        self._initialize_db()

    def _initialize_db(self):
        data = PersistenceManager.load_vector_data()
        if data:
            self.index, self.titles, self.urls, _ = data
        else:
            self.rebuild_index()

    def rebuild_index(self):
        bookmarks = get_firefox_bookmarks()
        if not bookmarks:
            self.titles, self.urls = [], []
            return # Handle empty gracefully
        
        self.titles = [bm["title"] for bm in bookmarks]
        self.urls = [bm["url"] for bm in bookmarks]
        embeddings = self.model.encode(self.titles, convert_to_numpy=True)
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        PersistenceManager.save_vector_data(self.index, self.titles, self.urls, embeddings)

    def search_optimized(self, query: str, limit: int, ignored_set: Set[str]) -> List[Tuple[int, float]]:
        """
        Performs an iterative search to find 'limit' valid results.
        Instead of loading ALL results, it fetches (limit + buffer) and expands if needed.
        """
        if not self.index or len(self.titles) == 0:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        total_items = len(self.titles)
        
        valid_results = []
        # Optimization: Start with a buffer. If user wants 10, fetch 20.
        # This handles cases where some top items are ignored.
        k = min(limit + len(ignored_set) + 10, total_items)
        
        while len(valid_results) < limit and k <= total_items:
            distances, indices = self.index.search(query_vec, k)
            
            # Reset valid results to ensure order is correct on re-expansion
            temp_results = []
            
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue # FAISS padding
                
                url = self.urls[idx]
                if url not in ignored_set:
                    temp_results.append((idx, distances[0][i]))
            
            valid_results = temp_results
            
            # Break if we have enough, otherwise double K and retry
            if len(valid_results) >= limit or k == total_items:
                break
            
            # Geometric expansion
            k = min(k * 2, total_items)

        return valid_results[:limit]

    def get_bookmark(self, idx: int) -> Tuple[str, str]:
        return self.titles[idx], self.urls[idx]

    def get_title_map(self) -> Dict[str, str]:
        return dict(zip(self.urls, self.titles))

    # Proxy methods for DatabaseHandler
    def get_ignored_set(self) -> Set[str]:
        return self.db_handler.load_ignored_set()

    def toggle_ignore(self, url: str, is_currently_ignored: bool):
        if is_currently_ignored:
            self.db_handler.remove_ignored(url)
        else:
            self.db_handler.add_ignored(url)

# =========================================================
# 4. PRESENTATION LAYER
# =========================================================
class UIManager:
    def __init__(self, engine: SearchEngine):
        self.engine = engine
        self._init_session_state()

    def _init_session_state(self):
        # We load the ignored set into session state to avoid hitting DB on every tiny re-render
        if "ignored_cache" not in st.session_state:
            st.session_state.ignored_cache = self.engine.get_ignored_set()
            
        if "results_limit" not in st.session_state:
            st.session_state.results_limit = AppConfig.DEFAULT_RESULTS_LIMIT
            
        if "ignored_list_limit" not in st.session_state:
            st.session_state.ignored_list_limit = AppConfig.DEFAULT_IGNORED_LIMIT

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
        """Optimized toggle: Updates DB and Session Cache immediately."""
        is_ignored = url in st.session_state.ignored_cache
        
        # 1. Update Persistence (SQLite)
        self.engine.toggle_ignore(url, is_ignored)
        
        # 2. Update In-Memory Cache (Immediate UI feedback)
        if is_ignored:
            st.session_state.ignored_cache.remove(url)
        else:
            st.session_state.ignored_cache.add(url)

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
            st.info(f"Ignored Items: {len(st.session_state.ignored_cache)}")

    def render_search_results(self, query: str):
        # Use Optimized Search
        results = self.engine.search_optimized(
            query, 
            # We request slightly more than the limit to determine if "Load More" is needed
            limit=st.session_state.results_limit + 1, 
            ignored_set=st.session_state.ignored_cache
        )
        
        has_more = len(results) > st.session_state.results_limit
        visible_results = results[:st.session_state.results_limit]

        st.subheader(f"Matches")
        
        for idx, dist in visible_results:
            title, url = self.engine.get_bookmark(idx)
            
            c1, c2 = st.columns([0.85, 0.15])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(f"{url} | Score: {dist:.2f}")
            with c2:
                # Key must be unique per render
                st.toggle(
                    "Ignore", 
                    value=False, # Search results are by definition NOT ignored
                    key=f"search_tog_{idx}", 
                    on_change=self._handle_toggle, 
                    args=(url,)
                )
            st.divider()

        # Load More Button
        if has_more:
            if st.button("🔽 Load More Results"):
                st.session_state.results_limit += 10
                st.rerun()

        if visible_results and st.button("📊 Visualize These Results"):
            titles = [self.engine.titles[i] for i, _ in visible_results]
            self._render_wordcloud(titles, "Search Context")

    def render_ignored_list(self):
        ignored = list(st.session_state.ignored_cache)
        if not ignored:
            return

        st.markdown("---")
        with st.expander(f"🚫 Ignored Items ({len(ignored)})"):
            visible_ignored = ignored[:st.session_state.ignored_list_limit]
            title_map = self.engine.get_title_map()

            for url in visible_ignored:
                title = title_map.get(url, "Unknown Title")
                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    st.write(f"**{title}**")
                    st.caption(url)
                with c2:
                    st.toggle(
                        "Ignore", 
                        value=True, # Ignored items are by definition ignored
                        key=f"ign_{abs(hash(url))}",
                        on_change=self._handle_toggle, 
                        args=(url,)
                    )
                st.divider()

            if len(ignored) > st.session_state.ignored_list_limit:
                if st.button("🔽 Load More Ignored"):
                    st.session_state.ignored_list_limit += 5
                    st.rerun()

    def render_main(self):
        st.title("🔍 Bookmark Smart Search (Optimized)")
        self.render_sidebar()
        
        query = st.text_input("Search bookmarks...")
        if query:
            self.render_search_results(query)
        
        self.render_ignored_list()

# =========================================================
# 5. ORCHESTRATION
# =========================================================
@st.cache_resource
def get_engine():
    return SearchEngine()

def main():
    try:
        engine = get_engine()
        ui = UIManager(engine)
        ui.render_main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        # Option to clear cache if things go wrong
        if st.button("Hard Reset"):
            AppConfig.clear_cache()
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()