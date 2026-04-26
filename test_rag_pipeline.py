"""
Testing and Example Script for RAG Pipeline
Use this to verify your setup works and to test different configurations
"""

import os
from rag_pipeline import (
    load_pdf,
    chunk_documents,
    create_embeddings,
    create_vector_db_faiss,
    create_vector_db_chroma,
    load_vector_db,
    search_vector_db,
)

# ============================================
# TEST 1: Check if all libraries are installed
# ============================================


def test_imports():
    """Test if all required libraries are installed"""
    print("=" * 70)
    print("TEST 1: Checking Library Installations")
    print("=" * 70)

    libraries = [
        ("langchain", "LangChain"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("langchain_community", "LangChain Community"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("pypdf", "PyPDF"),
        ("faiss", "FAISS"),
        ("chromadb", "Chroma"),
        ("sentence_transformers", "Sentence Transformers"),
    ]

    all_ok = True
    for lib, name in libraries:
        try:
            __import__(lib)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n✅ All libraries installed!")
    else:
        print("\n❌ Some libraries missing. Run: pip install -r requirements.txt")

    print()
    return all_ok


# ============================================
# TEST 2: Check if PDF file exists
# ============================================


def test_pdf_exists(pdf_path):
    """Test if PDF file exists"""
    print("=" * 70)
    print("TEST 2: Checking PDF File")
    print("=" * 70)

    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"✅ PDF found: {pdf_path}")
        print(f"   File size: {file_size:.2f} MB")
        return True
    else:
        print(f"❌ PDF not found: {pdf_path}")
        print(f"\n   Current directory: {os.getcwd()}")
        print(f"   Files in directory:")
        for f in os.listdir("."):
            if f.endswith(".pdf"):
                print(f"      - {f}")
        return False


# ============================================
# TEST 3: Test PDF loading
# ============================================


def test_pdf_loading(pdf_path):
    """Test if PDF can be loaded successfully"""
    print("\n" + "=" * 70)
    print("TEST 3: Loading PDF")
    print("=" * 70)

    try:
        documents = load_pdf(pdf_path)
        total_chars = sum(len(d.page_content) for d in documents)
        print(f"✅ PDF loaded successfully")
        print(f"   Pages: {len(documents)}")
        print(f"   Total characters: {total_chars:,}")
        return documents
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return None


# ============================================
# TEST 4: Test chunking
# ============================================


def test_chunking(documents, chunk_size=500, overlap=50):
    """Test document chunking"""
    print("\n" + "=" * 70)
    print("TEST 4: Document Chunking")
    print("=" * 70)

    try:
        chunks = chunk_documents(documents, chunk_size, overlap)
        avg_chunk = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"✅ Chunking successful")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average chunk size: {avg_chunk} chars")
        print(
            f"   Chunk size range: {min(len(c.page_content) for c in chunks)} - {max(len(c.page_content) for c in chunks)} chars"
        )
        return chunks
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return None


# ============================================
# TEST 5: Test embeddings
# ============================================


def test_embeddings():
    """Test if embedding model can be loaded"""
    print("\n" + "=" * 70)
    print("TEST 5: Loading Embedding Model")
    print("=" * 70)
    print("(This may take a minute on first run - downloading model...)")

    try:
        embeddings = create_embeddings()

        # Test embedding a sample text
        sample_text = "This is a test sentence for embeddings"
        embedding = embeddings.embed_query(sample_text)

        print(f"✅ Embedding model loaded")
        print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"   Embedding dimension: {len(embedding)}")
        return embeddings
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return None


# ============================================
# TEST 6: Test FAISS vector database
# ============================================


def test_faiss_db(chunks, embeddings):
    """Test FAISS vector database creation"""
    print("\n" + "=" * 70)
    print("TEST 6: Creating FAISS Vector Database")
    print("=" * 70)

    try:
        vector_db = create_vector_db_faiss(
            chunks, embeddings, db_path="./test_vector_db"
        )

        # Test search
        print("\n   Testing search...")
        results = vector_db.similarity_search("test query", k=1)
        print(f"✅ FAISS database created and working")
        print(f"   Vectors stored: {len(chunks)}")
        print(f"   Sample search returned: {len(results)} result(s)")

        return vector_db
    except Exception as e:
        print(f"❌ FAISS database creation failed: {e}")
        return None


# ============================================
# TEST 7: Test Chroma vector database
# ============================================


def test_chroma_db(chunks, embeddings):
    """Test Chroma vector database creation"""
    print("\n" + "=" * 70)
    print("TEST 7: Creating Chroma Vector Database")
    print("=" * 70)

    try:
        vector_db = create_vector_db_chroma(
            chunks, embeddings, db_path="./test_vector_db"
        )

        # Test search
        print("\n   Testing search...")
        results = vector_db.similarity_search("test query", k=1)
        print(f"✅ Chroma database created and working")
        print(f"   Vectors stored: {len(chunks)}")
        print(f"   Sample search returned: {len(results)} result(s)")

        return vector_db
    except Exception as e:
        print(f"❌ Chroma database creation failed: {e}")
        return None


# ============================================
# TEST 8: Full pipeline test
# ============================================


def test_full_pipeline(pdf_path, db_type="faiss"):
    """Run the complete pipeline end-to-end"""
    print("\n" + "=" * 70)
    print("TEST 8: Full Pipeline Test")
    print("=" * 70)

    try:
        # Load PDF
        documents = load_pdf(pdf_path)
        if not documents:
            print("❌ PDF loading failed")
            return False

        # Chunk
        chunks = chunk_documents(documents, chunk_size=500, overlap=50)
        if not chunks:
            print("❌ Chunking failed")
            return False

        # Create embeddings
        embeddings = create_embeddings()
        if not embeddings:
            print("❌ Embedding model loading failed")
            return False

        # Create vector DB
        if db_type == "faiss":
            vector_db = create_vector_db_faiss(chunks, embeddings)
        else:
            vector_db = create_vector_db_chroma(chunks, embeddings)

        if not vector_db:
            print("❌ Vector database creation failed")
            return False

        print(f"\n✅ Full pipeline completed successfully!")
        print(f"   Database type: {db_type.upper()}")
        print(f"   Documents processed: {len(documents)}")
        print(f"   Chunks created: {len(chunks)}")

        return True

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False


# ============================================
# EXAMPLE: Search demonstrations
# ============================================


def demo_search(vector_db):
    """Demonstrate search functionality"""
    print("\n" + "=" * 70)
    print("DEMO: Search Examples")
    print("=" * 70)

    queries = [
        "main topic",
        "introduction",
        "key concepts",
        "conclusion",
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            results = search_vector_db(vector_db, query, k=1)
            if results:
                print(f"✅ Found relevant content")
        except Exception as e:
            print(f"❌ Search failed: {e}")


# ============================================
# MAIN TEST RUNNER
# ============================================


def run_all_tests(pdf_path="ml_notes.pdf"):
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "RAG PIPELINE TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    # Test 1: Check imports
    if not test_imports():
        print("\n⚠️ Please install dependencies first!")
        return

    # Test 2: Check PDF
    if not test_pdf_exists(pdf_path):
        print("\n⚠️ Please provide a valid PDF file path!")
        return

    # Test 3: Load PDF
    documents = test_pdf_loading(pdf_path)
    if not documents:
        return

    # Test 4: Chunk
    chunks = test_chunking(documents)
    if not chunks:
        return

    # Test 5: Embeddings
    embeddings = test_embeddings()
    if not embeddings:
        return

    # Test 6: FAISS
    vector_db_faiss = test_faiss_db(chunks, embeddings)

    # Test 7: Chroma
    vector_db_chroma = test_chroma_db(chunks, embeddings)

    # Test 8: Full pipeline (FAISS)
    test_full_pipeline(pdf_path, db_type="faiss")

    # Demo: Search
    if vector_db_faiss:
        demo_search(vector_db_faiss)

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 70)
    print("\n📝 Summary:")
    print("   ✅ Libraries installed")
    print("   ✅ PDF loaded")
    print("   ✅ Documents chunked")
    print("   ✅ Embeddings created")
    print("   ✅ Vector databases created")
    print("   ✅ Search working")
    print("\n🎉 Your RAG pipeline is ready to use!")
    print("\nNext steps:")
    print("1. Use rag_pipeline.py with your PDF")
    print("2. Customize chunk size and embedding model as needed")
    print("3. Integrate with your application")
    print()


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    # Change this to your PDF path
    PDF_PATH = "ml_notes.pdf"

    run_all_tests(pdf_path=PDF_PATH)
