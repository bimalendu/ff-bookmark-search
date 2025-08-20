# 🔍 Firefox Bookmark Semantic Search

A Streamlit-powered app to **search your Firefox bookmarks using semantic similarity**.  
Bookmark titles are converted into vectors using a transformer model and stored in a persistent FAISS index.

---

## 🚀 Features

- 🔎 **Semantic search**: Search bookmarks using natural language.
- 📥 **Auto-detects Firefox profile** (Windows, macOS, Linux).
- 🔒 **Safe read**: Copies `places.sqlite` to avoid locking conflicts.
- 💾 **Persistent FAISS index**: No need to re-index every time.
- 🧠 **Embeddings with SentenceTransformers**: Uses `all-MiniLM-L6-v2` for fast and accurate vector generation.

---

## 📦 Dependencies

Make sure you have Python 3.8 or newer.

Install required packages with uv:

```uv
uv init
uv add streamlit sentence-transformers faiss-cpu
uv run -- streamlit run app.py