"""
RAG Pipeline: PDF → Chunks → Embeddings → Vector DB
Complete implementation for HexaWar GenAI Internship

Author: Mahi Bhosale
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

# Import necessary libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION - Change these as needed
# ============================================

PDF_FILE_PATH = "ml_notes.pdf"   # Path to your PDF file
CHUNK_SIZE = 500                  # Size of each chunk in characters
CHUNK_OVERLAP = 50                # Overlap between chunks for context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast & accurate
DB_TYPE = "faiss"                 # Choose: "faiss" or "chroma"
DB_FOLDER = "./vector_db"         # Where to save the vector database


# ============================================
# STEP 1: Load PDF and Extract Text
# ============================================

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF file and extract text page by page.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Document objects, one per page
    
    Raises:
        FileNotFoundError: If PDF does not exist at given path
    """
    print(f"\n📄 Loading PDF from: {pdf_path}")
    logger.info(f"Loading PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    total_chars = sum(len(d.page_content) for d in documents)
    print(f"✅ Loaded {len(documents)} pages | {total_chars:,} total characters")

    for i, doc in enumerate(documents):
        print(f"   Page {i+1}: {len(doc.page_content)} characters")

    logger.info(f"Loaded {len(documents)} pages from PDF")
    return documents


# ============================================
# STEP 2: Perform Recursive Character Chunking
# ============================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    How it works:
    - Tries to split at: paragraphs → lines → words → characters
    - Each chunk has max size = chunk_size characters
    - Consecutive chunks share 'overlap' characters for context continuity

    Args:
        documents: List of Document objects (one per page)
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters shared between consecutive chunks

    Returns:
        List of chunked Document objects
    """
    print(f"\n✂️  Chunking documents...")
    print(f"   Strategy: RecursiveCharacterTextSplitter")
    print(f"   Chunk size: {chunk_size} chars | Overlap: {overlap} chars")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  # Priority order for splitting
    )

    chunks = splitter.split_documents(documents)

    lengths = [len(c.page_content) for c in chunks]
    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Min: {min(lengths)} chars | Max: {max(lengths)} chars | Avg: {sum(lengths)//len(lengths)} chars")

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# ============================================
# STEP 3: Create Embeddings using HuggingFace
# ============================================

def create_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace sentence-transformer embedding model.

    Converts text chunks into dense numerical vectors (embeddings).
    Similar text → similar vectors → enables semantic search.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HuggingFaceEmbeddings object ready for encoding
    """
    print(f"\n🤖 Loading embedding model: {model_name}")
    print(f"   (First run downloads ~90 MB — cached for future runs)")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Verify model works with a test query
    test_vector = embeddings.embed_query("test sentence")
    print(f"✅ Embedding model loaded | Vector dimension: {len(test_vector)}")
    logger.info(f"Embeddings loaded: {model_name} | Dim: {len(test_vector)}")

    return embeddings


# ============================================
# STEP 4a: Store in FAISS Vector Database
# ============================================

def create_vector_db_faiss(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    db_path: str = DB_FOLDER
) -> FAISS:
    """
    Create and persist a FAISS vector database from document chunks.

    FAISS (Facebook AI Similarity Search):
    - Extremely fast similarity search
    - Stores vectors locally on disk
    - Ideal for datasets up to millions of vectors

    Args:
        chunks: List of chunked Document objects
        embeddings: Loaded HuggingFace embedding model
        db_path: Directory path to save FAISS index

    Returns:
        FAISS vector store object
    """
    print(f"\n💾 Creating FAISS vector database...")
    print(f"   Embedding {len(chunks)} chunks into vectors...")

    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)

    os.makedirs(db_path, exist_ok=True)
    save_path = f"{db_path}/faiss_index"
    vector_db.save_local(save_path)

    print(f"✅ FAISS database saved to: {save_path}")
    print(f"   Files: index.faiss (vectors) + index.pkl (metadata)")
    print(f"   Total vectors stored: {len(chunks)}")

    logger.info(f"FAISS DB created with {len(chunks)} vectors at {save_path}")
    return vector_db


# ============================================
# STEP 4b: Store in Chroma Vector Database
# ============================================

def create_vector_db_chroma(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    db_path: str = DB_FOLDER
) -> Chroma:
    """
    Create and persist a Chroma vector database from document chunks.

    Chroma:
    - Open-source vector database
    - Better metadata filtering than FAISS
    - Good for larger, more complex datasets

    Args:
        chunks: List of chunked Document objects
        embeddings: Loaded HuggingFace embedding model
        db_path: Directory path to save Chroma database

    Returns:
        Chroma vector store object
    """
    print(f"\n💾 Creating Chroma vector database...")
    print(f"   Embedding {len(chunks)} chunks into vectors...")

    os.makedirs(db_path, exist_ok=True)
    save_path = f"{db_path}/chroma_db"

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=save_path
    )

    print(f"✅ Chroma database saved to: {save_path}")
    print(f"   Total vectors stored: {len(chunks)}")

    logger.info(f"Chroma DB created with {len(chunks)} vectors at {save_path}")
    return vector_db


# ============================================
# STEP 5: Load Existing Vector Database
# ============================================

def load_vector_db(
    db_type: str = DB_TYPE,
    embeddings: HuggingFaceEmbeddings = None,
    db_path: str = DB_FOLDER
):
    """
    Load a previously saved vector database from disk.

    Args:
        db_type: Type of database — "faiss" or "chroma"
        embeddings: Embedding model (auto-loaded if None)
        db_path: Directory where database was saved

    Returns:
        Vector store object (FAISS or Chroma)

    Raises:
        ValueError: If db_type is not "faiss" or "chroma"
    """
    if embeddings is None:
        embeddings = create_embeddings()

    if db_type == "faiss":
        load_path = f"{db_path}/faiss_index"
        print(f"📂 Loading FAISS database from: {load_path}")
        vector_db = FAISS.load_local(
            load_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    elif db_type == "chroma":
        load_path = f"{db_path}/chroma_db"
        print(f"📂 Loading Chroma database from: {load_path}")
        vector_db = Chroma(
            persist_directory=load_path,
            embedding_function=embeddings
        )
    else:
        raise ValueError(f"Unknown db_type: '{db_type}'. Use 'faiss' or 'chroma'.")

    print(f"✅ Vector database loaded successfully")
    logger.info(f"Loaded {db_type} database from {db_path}")
    return vector_db


# ============================================
# STEP 6: Search and Query Vector Database
# ============================================

def search_vector_db(vector_db, query: str, k: int = 3) -> list:
    """
    Search vector database for chunks semantically similar to the query.

    Uses cosine similarity between query vector and stored vectors.
    Returns the top-k most relevant chunks with similarity scores.

    Args:
        vector_db: FAISS or Chroma vector store object
        query: Natural language search query
        k: Number of top results to return

    Returns:
        List of (Document, score) tuples sorted by relevance
    """
    print(f"\n🔍 Searching: '{query}'")

    # Use similarity search with scores for richer output
    try:
        results_with_scores = vector_db.similarity_search_with_score(query, k=k)
        print(f"✅ Found {len(results_with_scores)} results:\n")

        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"--- Result {i} | Page: {doc.metadata.get('page', 'N/A')} | Score: {score:.4f} ---")
            print(f"{doc.page_content[:300]}...")
            print()

        return results_with_scores

    except Exception:
        # Fallback: basic search without scores (Chroma compatibility)
        results = vector_db.similarity_search(query, k=k)
        print(f"✅ Found {len(results)} results:\n")

        for i, doc in enumerate(results, 1):
            print(f"--- Result {i} | Page: {doc.metadata.get('page', 'N/A')} ---")
            print(f"{doc.page_content[:300]}...")
            print()

        return results


# ============================================
# COMPLETE PIPELINE RUNNER
# ============================================

def run_complete_pipeline(
    pdf_path: str,
    db_type: str = DB_TYPE,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Tuple:
    """
    Execute the complete RAG indexing pipeline end-to-end.

    Pipeline stages:
        1. Load PDF → extract text from all pages
        2. Chunk text → split into overlapping segments
        3. Create embeddings → convert chunks to vectors
        4. Store in vector DB → persist FAISS or Chroma index

    Args:
        pdf_path: Path to input PDF file
        db_type: Vector database type — "faiss" or "chroma"
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between consecutive chunks

    Returns:
        Tuple of (vector_db, embeddings)
    """
    print("=" * 70)
    print("🚀 RAG PIPELINE — HexaWar GenAI Internship")
    print("=" * 70)
    print(f"   PDF: {pdf_path}")
    print(f"   Chunk size: {chunk_size} | Overlap: {overlap}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   Vector DB: {db_type.upper()}")
    print("=" * 70)

    # Step 1: Load PDF
    documents = load_pdf(pdf_path)

    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size, overlap)

    # Step 3: Create embeddings
    embeddings = create_embeddings()

    # Step 4: Create and save vector database
    if db_type == "faiss":
        vector_db = create_vector_db_faiss(chunks, embeddings)
    elif db_type == "chroma":
        vector_db = create_vector_db_chroma(chunks, embeddings)
    else:
        raise ValueError(f"Unknown db_type: '{db_type}'. Use 'faiss' or 'chroma'.")

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print(f"   Pages loaded   : {len(documents)}")
    print(f"   Chunks created : {len(chunks)}")
    print(f"   Vectors stored : {len(chunks)}")
    print(f"   DB saved to    : {DB_FOLDER}/{db_type}_index")
    print("=" * 70)

    return vector_db, embeddings


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":

    try:
        # Run the complete pipeline
        vector_db, embeddings = run_complete_pipeline(
            pdf_path=PDF_FILE_PATH,
            db_type=DB_TYPE,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        # Test the vector database with sample queries
        print("\n" + "=" * 70)
        print("🔍 TESTING VECTOR DATABASE — Sample Queries")
        print("=" * 70)

        test_queries = [
            "What is the main topic of this document?",
            "Explain the key machine learning concepts",
            "What is gradient descent?",
        ]

        for query in test_queries:
            try:
                search_vector_db(vector_db, query, k=2)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                print(f"⚠️  Search failed: {e}\n")

        print("=" * 70)
        print("✅ All tests completed successfully!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        print(f"\n📝 Fix: Update PDF_FILE_PATH at the top of this script:")
        print(f"   PDF_FILE_PATH = 'your_actual_file.pdf'")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Verify your PDF file exists at the specified path")
        print("   3. Check your internet connection (for model download)")
        print("   4. Run: python test_rag_pipeline.py for detailed diagnostics")
