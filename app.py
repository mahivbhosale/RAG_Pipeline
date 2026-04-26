"""
app.py — Streamlit UI for RAG Pipeline
HexaWar GenAI Internship

Author: Mahi Bhosale
Date: 2025

Run with: streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    load_pdf,
    chunk_documents,
    create_embeddings,
    create_vector_db_faiss,
    create_vector_db_chroma,
    load_vector_db,
    search_vector_db,
    EMBEDDING_MODEL,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline — HexaWar",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .step-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    .step-number {
        font-family: 'JetBrains Mono', monospace;
        color: #00ff88;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .result-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 4px solid #00ff88;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .score-badge {
        background: #00ff8820;
        color: #00ff88;
        border: 1px solid #00ff8840;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .page-badge {
        background: #00d4ff20;
        color: #00d4ff;
        border: 1px solid #00d4ff40;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-box {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #00ff88;
        display: block;
    }

    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .pipeline-flow {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #888;
    }

    .flow-step {
        color: #00ff88;
        font-weight: 600;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
    }

    .stTextInput > div > div > input {
        font-family: 'JetBrains Mono', monospace;
        background: #0d1117;
        border: 1px solid #21262d;
        color: white;
    }

    div[data-testid="stSelectbox"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "vector_db": None,
        "embeddings": None,
        "pipeline_stats": {},
        "search_history": [],
        "db_type": "faiss",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 RAG Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">PDF → Chunks → Embeddings → Vector DB → Search</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    chunk_size = st.slider(
        "Chunk Size (chars)",
        min_value=200, max_value=1000, value=500, step=50,
        help="Size of each text chunk in characters"
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (chars)",
        min_value=0, max_value=200, value=50, step=10,
        help="Overlap between consecutive chunks"
    )

    db_type = st.selectbox(
        "Vector Database",
        options=["faiss", "chroma"],
        help="FAISS = faster | Chroma = more features"
    )
    st.session_state.db_type = db_type

    top_k = st.slider(
        "Search Results (k)",
        min_value=1, max_value=10, value=3,
        help="Number of similar chunks to return"
    )

    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")

    if st.session_state.vector_db is not None:
        stats = st.session_state.pipeline_stats
        st.success("✅ Database Ready")
        st.markdown(f"""
        <div class="pipeline-flow">
        <span class="flow-step">Pages:</span> {stats.get('pages', '-')}<br>
        <span class="flow-step">Chunks:</span> {stats.get('chunks', '-')}<br>
        <span class="flow-step">Vectors:</span> {stats.get('chunks', '-')}<br>
        <span class="flow-step">Dim:</span> 384<br>
        <span class="flow-step">DB:</span> {stats.get('db_type', '-').upper()}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No database loaded")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #888;">
    Model: all-MiniLM-L6-v2<br>
    Framework: LangChain<br>
    Author: Mahi Bhosale<br>
    HexaWar GenAI Internship
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Reset Pipeline", use_container_width=True):
        st.session_state.vector_db = None
        st.session_state.embeddings = None
        st.session_state.pipeline_stats = {}
        st.session_state.search_history = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📄 Build Index", "🔍 Search", "📜 History"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — BUILD INDEX
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Build Vector Database")
    st.markdown("Upload a PDF and build a searchable vector index from it.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload any PDF — research paper, notes, report, textbook"
        )

        if uploaded_file:
            st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    with col2:
        st.markdown("#### Pipeline Flow")
        st.markdown("""
        <div class="pipeline-flow">
        <span class="flow-step">1.</span> Load PDF<br>
        <span class="flow-step">2.</span> Chunk text<br>
        <span class="flow-step">3.</span> Embed chunks<br>
        <span class="flow-step">4.</span> Store vectors<br>
        <span class="flow-step">5.</span> Ready to search!
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Run RAG Pipeline", type="primary", use_container_width=True, disabled=not uploaded_file):

        # Save uploaded file
        upload_dir = "/tmp/rag_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress = st.progress(0)
        status = st.empty()

        try:
            # Step 1: Load PDF
            status.info("📄 Step 1/4 — Loading PDF...")
            progress.progress(10)
            documents = load_pdf(pdf_path)
            progress.progress(25)
            st.success(f"✅ Loaded **{len(documents)} pages**")

            # Step 2: Chunk
            status.info("✂️ Step 2/4 — Chunking text...")
            progress.progress(35)
            chunks = chunk_documents(documents, chunk_size, chunk_overlap)
            progress.progress(50)
            st.success(f"✅ Created **{len(chunks)} chunks** ({chunk_size} chars, {chunk_overlap} overlap)")

            # Step 3: Embeddings
            status.info("🤖 Step 3/4 — Loading embedding model...")
            progress.progress(55)
            if st.session_state.embeddings is None:
                embeddings = create_embeddings()
                st.session_state.embeddings = embeddings
            else:
                embeddings = st.session_state.embeddings
            progress.progress(75)
            st.success(f"✅ Embedding model ready (384 dimensions)")

            # Step 4: Vector DB
            status.info(f"💾 Step 4/4 — Building {db_type.upper()} vector database...")
            progress.progress(80)
            if db_type == "faiss":
                vector_db = create_vector_db_faiss(chunks, embeddings)
            else:
                vector_db = create_vector_db_chroma(chunks, embeddings)

            progress.progress(100)
            st.session_state.vector_db = vector_db
            st.session_state.pipeline_stats = {
                "pages": len(documents),
                "chunks": len(chunks),
                "db_type": db_type,
                "pdf_name": uploaded_file.name,
            }

            status.empty()
            progress.empty()

            # Show stats
            st.markdown("---")
            st.markdown("### ✅ Pipeline Complete!")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="stat-box">
                    <span class="stat-value">{len(documents)}</span>
                    <span class="stat-label">Pages Loaded</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-box">
                    <span class="stat-value">{len(chunks)}</span>
                    <span class="stat-label">Chunks Created</span>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-box">
                    <span class="stat-value">384</span>
                    <span class="stat-label">Vector Dims</span>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="stat-box">
                    <span class="stat-value">{db_type.upper()}</span>
                    <span class="stat-label">Database Type</span>
                </div>""", unsafe_allow_html=True)

            st.info("👉 Go to the **Search** tab to query your document!")

        except Exception as e:
            status.empty()
            progress.empty()
            st.error(f"❌ Pipeline failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Semantic Search")
    st.markdown("Ask any question — the system finds the most relevant chunks from your document.")

    if st.session_state.vector_db is None:
        st.warning("⚠️ No vector database loaded. Please build an index in the **Build Index** tab first.")
    else:
        stats = st.session_state.pipeline_stats
        st.info(f"🗄️ Database ready — **{stats.get('pdf_name', 'document')}** | {stats.get('chunks', 0)} vectors | {stats.get('db_type', 'faiss').upper()}")

        query = st.text_input(
            "Enter your search query",
            placeholder="e.g. What is gradient descent? | Explain decision trees | What is regularization?",
            label_visibility="collapsed"
        )

        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True, disabled=not query.strip())
        with col_clear:
            if st.button("Clear History", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()

        # Example queries
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(3)
        examples = [
            "What is gradient descent?",
            "Explain decision trees",
            "What is regularization?",
            "How does KNN work?",
            "What is random forest?",
            "Explain logistic regression",
        ]
        for i, ex in enumerate(examples):
            with ex_cols[i % 3]:
                if st.button(f"💬 {ex}", key=f"ex_{i}", use_container_width=True):
                    query = ex
                    search_btn = True

        if search_btn and query.strip():
            with st.spinner("🔍 Searching vector database..."):
                try:
                    start_time = time.time()
                    results = st.session_state.vector_db.similarity_search_with_score(
                        query, k=top_k
                    )
                    elapsed = time.time() - start_time

                    # Save to history
                    st.session_state.search_history.append({
                        "query": query,
                        "results": results,
                        "time": elapsed,
                        "k": top_k
                    })

                    st.markdown(f"---")
                    st.markdown(f"**Results for:** `{query}` — found in `{elapsed:.3f}s`")
                    st.markdown("")

                    for i, (doc, score) in enumerate(results, 1):
                        page = doc.metadata.get('page', 'N/A')
                        # Convert distance to similarity (lower distance = higher similarity)
                        similarity = max(0, 1 - score) if score > 0 else 1.0

                        st.markdown(f"""
                        <div class="result-card">
                            <div style="margin-bottom: 0.5rem;">
                                <span style="color: #00ff88; font-weight: 600;">Result {i}</span>
                                &nbsp;&nbsp;
                                <span class="page-badge">Page {page}</span>
                                &nbsp;&nbsp;
                                <span class="score-badge">Score: {score:.4f}</span>
                            </div>
                            <div style="color: #ccc; line-height: 1.6;">
                                {doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Search failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Search History")

    if not st.session_state.search_history:
        st.info("No searches yet. Go to the **Search** tab and try some queries!")
    else:
        st.markdown(f"**{len(st.session_state.search_history)} searches** this session")
        st.markdown("---")

        for i, item in enumerate(reversed(st.session_state.search_history), 1):
            with st.expander(f"🔍 {item['query']} — {item['time']:.3f}s — {item['k']} results"):
                for j, (doc, score) in enumerate(item['results'], 1):
                    page = doc.metadata.get('page', 'N/A')
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="margin-bottom: 0.5rem;">
                            <span style="color: #00ff88;">Result {j}</span>
                            &nbsp;
                            <span class="page-badge">Page {page}</span>
                            &nbsp;
                            <span class="score-badge">Score: {score:.4f}</span>
                        </div>
                        <div style="color: #ccc;">{doc.page_content[:300]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
