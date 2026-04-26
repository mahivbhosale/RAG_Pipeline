# Complete Guide: PDF to Vector DB (RAG Pipeline)
## HexaWar GenAI Internship Task

---

## 📋 Overview

This guide will help you:
1. Install all required libraries
2. Load and read a PDF file
3. Perform recursive character chunking
4. Convert chunks to embeddings using HuggingFace
5. Store in FAISS or Chroma vector database
6. Test and query your vector DB

---

## 🛠️ Step 1: Install Required Libraries

Run these commands in your terminal:

```bash
pip install --upgrade pip

# Core dependencies
pip install langchain langchain-community langchain-text-splitters

# PDF processing
pip install pypdf pdfplumber

# Vector databases
pip install faiss-cpu chroma

# Embeddings from HuggingFace
pip install sentence-transformers huggingface-hub

# Utilities
pip install python-dotenv
```

**What each does:**
- `langchain` - Framework for building RAG pipelines
- `pypdf` - Read PDF files
- `sentence-transformers` - Create embeddings (HuggingFace models)
- `faiss-cpu` - Store and search vectors efficiently
- `chroma` - Alternative vector database

---

## 🎯 Step 2: Understand Recursive Character Chunking

**What it does:**
- Splits text into chunks of size ~500 characters (configurable)
- Tries to split at natural boundaries (paragraphs, sentences, words)
- Adds overlap (e.g., 50 chars) between chunks for context

**Example:**
```
Original text (5000 chars) 
    ↓
Chunk 1: "Introduction... [0-500 chars]"
Chunk 2: "Introduction continued... [450-950 chars]"  ← 50 char overlap
Chunk 3: "... [900-1400 chars]"
... and so on
```

---

## 💻 Step 3: Complete Python Code

### File: `rag_pipeline.py`

```python
"""
RAG Pipeline: PDF → Chunks → Embeddings → Vector DB
Author: Your Name
Date: 2024
"""

import os
from pathlib import Path
from typing import List

# Import necessary libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================
# CONFIGURATION - Change these as needed
# ============================================

PDF_FILE_PATH = "your_document.pdf"  # Path to your PDF
CHUNK_SIZE = 500  # Size of each chunk in characters
CHUNK_OVERLAP = 50  # Overlap between chunks for context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast model
DB_TYPE = "faiss"  # Choose: "faiss" or "chroma"
DB_FOLDER = "./vector_db"  # Where to save the vector database

# ============================================
# STEP 1: Load PDF and Extract Text
# ============================================

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF file and extract text.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects with page content
    """
    print(f"📄 Loading PDF from: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} pages from PDF")
    return documents


# ============================================
# STEP 2: Perform Recursive Character Chunking
# ============================================

def chunk_documents(documents: List[Document], 
                   chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk (characters)
        overlap: Overlap between chunks (characters)
        
    Returns:
        List of chunked Document objects
    """
    print(f"\n🔪 Chunking documents...")
    print(f"   Chunk size: {chunk_size} chars")
    print(f"   Overlap: {overlap} chars")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]  # Try these in order
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Sample chunk length: {len(chunks[0].page_content)} chars")
    
    return chunks


# ============================================
# STEP 3: Create Embeddings using HuggingFace
# ============================================

def create_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Initialize HuggingFace embedding model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        HuggingFaceEmbeddings object
    """
    print(f"\n🤖 Loading embedding model: {model_name}")
    print(f"   (First time will download ~100-500 MB)")
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print(f"✅ Embedding model loaded successfully")
    return embeddings


# ============================================
# STEP 4: Store in Vector Database
# ============================================

def create_vector_db_faiss(chunks: List[Document], 
                          embeddings, 
                          db_path: str = DB_FOLDER) -> FAISS:
    """
    Create and save FAISS vector database.
    
    Args:
        chunks: List of chunked documents
        embeddings: Embedding model
        db_path: Path to save FAISS database
        
    Returns:
        FAISS vector store object
    """
    print(f"\n📦 Creating FAISS vector database...")
    
    # Create FAISS index from documents
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save to disk
    os.makedirs(db_path, exist_ok=True)
    vector_db.save_local(f"{db_path}/faiss_index")
    
    print(f"✅ FAISS database saved to: {db_path}/faiss_index")
    return vector_db


def create_vector_db_chroma(chunks: List[Document], 
                           embeddings, 
                           db_path: str = DB_FOLDER) -> Chroma:
    """
    Create and save Chroma vector database.
    
    Args:
        chunks: List of chunked documents
        embeddings: Embedding model
        db_path: Path to save Chroma database
        
    Returns:
        Chroma vector store object
    """
    print(f"\n📦 Creating Chroma vector database...")
    
    os.makedirs(db_path, exist_ok=True)
    
    # Create Chroma database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"{db_path}/chroma_db"
    )
    
    print(f"✅ Chroma database saved to: {db_path}/chroma_db")
    return vector_db


# ============================================
# STEP 5: Load Existing Vector Database
# ============================================

def load_vector_db(db_type: str = DB_TYPE, 
                  embeddings = None, 
                  db_path: str = DB_FOLDER):
    """
    Load existing vector database from disk.
    
    Args:
        db_type: "faiss" or "chroma"
        embeddings: Embedding model
        db_path: Path to vector database
        
    Returns:
        Vector store object
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    if db_type == "faiss":
        print(f"📂 Loading FAISS database from: {db_path}/faiss_index")
        vector_db = FAISS.load_local(
            f"{db_path}/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    elif db_type == "chroma":
        print(f"📂 Loading Chroma database from: {db_path}/chroma_db")
        vector_db = Chroma(
            persist_directory=f"{db_path}/chroma_db",
            embedding_function=embeddings
        )
    else:
        raise ValueError(f"Unknown db_type: {db_type}")
    
    print(f"✅ Vector database loaded successfully")
    return vector_db


# ============================================
# STEP 6: Search and Query Vector Database
# ============================================

def search_vector_db(vector_db, query: str, k: int = 3):
    """
    Search vector database for similar chunks.
    
    Args:
        vector_db: Vector store object
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of most similar documents
    """
    print(f"\n🔍 Searching for: '{query}'")
    
    results = vector_db.similarity_search(query, k=k)
    
    print(f"✅ Found {len(results)} similar results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Similarity score) ---")
        print(f"Page: {result.metadata.get('page', 'N/A')}")
        print(f"Content: {result.page_content[:200]}...")
    
    return results


# ============================================
# MAIN PIPELINE FUNCTION
# ============================================

def run_complete_pipeline(pdf_path: str, 
                         db_type: str = DB_TYPE,
                         chunk_size: int = CHUNK_SIZE,
                         overlap: int = CHUNK_OVERLAP):
    """
    Run the complete RAG pipeline from start to finish.
    
    Args:
        pdf_path: Path to PDF file
        db_type: "faiss" or "chroma"
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
    """
    print("=" * 60)
    print("🚀 STARTING RAG PIPELINE")
    print("=" * 60)
    
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
        raise ValueError(f"Unknown db_type: {db_type}")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)
    
    return vector_db, embeddings


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Run the complete pipeline
    vector_db, embeddings = run_complete_pipeline(
        pdf_path=PDF_FILE_PATH,
        db_type=DB_TYPE,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    # Example: Search the vector database
    print("\n" + "=" * 60)
    print("🔍 TESTING VECTOR DATABASE")
    print("=" * 60)
    
    # Try some queries
    test_queries = [
        "What is the main topic?",
        "Explain the key concepts",
        "Tell me about the introduction"
    ]
    
    for query in test_queries:
        search_vector_db(vector_db, query, k=2)
        print()
```

---

## 📝 Step 4: Quick Start Guide

### Option A: Using Your Own PDF

1. **Save the code above as `rag_pipeline.py`**

2. **Edit the configuration section:**
   ```python
   PDF_FILE_PATH = "path/to/your/file.pdf"  # Change this
   DB_TYPE = "faiss"  # or "chroma"
   ```

3. **Run the script:**
   ```bash
   python rag_pipeline.py
   ```

### Option B: Test with Sample Data

```bash
# Create a simple test PDF (optional - for testing)
python -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('test.pdf')
c.drawString(100, 750, 'Sample Document')
c.drawString(100, 730, 'This is a test PDF.')
c.showPage()
c.save()
"

# Edit rag_pipeline.py to use test.pdf
# Then run: python rag_pipeline.py
```

---

## 🎓 Understanding the Code Flow

```
rag_pipeline.py
    ↓
1. load_pdf() 
   → Reads PDF, extracts text
   ↓
2. chunk_documents()
   → Splits text into 500-char chunks with 50-char overlap
   ↓
3. create_embeddings()
   → Loads HuggingFace embedding model
   ↓
4. create_vector_db_faiss() OR create_vector_db_chroma()
   → Converts chunks to vectors and stores them
   ↓
5. Vector DB saved on disk
   ↓
6. search_vector_db()
   → Query the database to find similar chunks
```

---

## 📊 Configuration Options

### Embedding Models (Fast → Accurate)
```python
# Fast (recommended for start)
"sentence-transformers/all-MiniLM-L6-v2"

# Better quality
"sentence-transformers/all-mpnet-base-v2"

# Larger (slower but most accurate)
"sentence-transformers/all-roberta-large-v1"
```

### Chunk Size Guidelines
```python
CHUNK_SIZE = 256   # Small - good for dense documents
CHUNK_SIZE = 512   # Medium - balanced (recommended)
CHUNK_SIZE = 1024  # Large - for sparse documents
```

---

## ❓ Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Reinstall all dependencies
pip install --upgrade langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu chroma
```

### Issue: "PDF not found"
- Make sure your PDF path is correct
- Use absolute path or check current directory
- Example: `/home/user/documents/myfile.pdf`

### Issue: "Out of Memory"
- Reduce CHUNK_SIZE
- Use a smaller embedding model
- Process one page at a time

### Issue: "Download slow"
- First run downloads the embedding model (~100-500 MB)
- This is normal, happens only once

---

## 🚀 Next Steps (After Completing Task)

1. **Add error handling** - Handle missing files, empty PDFs
2. **Add logging** - Track pipeline progress
3. **Batch processing** - Handle multiple PDFs
4. **Web interface** - Build a search UI
5. **LLM integration** - Use vector DB with ChatGPT for Q&A

---

## 📚 Additional Resources

- LangChain Docs: https://docs.langchain.com/
- HuggingFace Models: https://huggingface.co/models?library=sentence-transformers
- FAISS: https://github.com/facebookresearch/faiss
- Chroma: https://docs.trychroma.com/

---

**Good luck with your HexaWar internship! 🎉**
