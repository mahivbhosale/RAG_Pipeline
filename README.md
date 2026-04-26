# 🧠 RAG Pipeline — PDF to Vector Database

> **HexaWare GenAI Internship Task**
> 👤 Author: Mahi Bhosale
> 🔗 Repo: https://github.com/mahivbhosale/RAG_Pipeline

A complete **RAG (Retrieval-Augmented Generation) indexing pipeline** that transforms any PDF document into a searchable vector database using LangChain, HuggingFace, and FAISS/Chroma.

---

## 📋 What It Does

```
PDF Document
     ↓  Step 1: Load
42 pages of text
     ↓  Step 2: Chunk
168 text pieces (500 chars, 50 overlap)
     ↓  Step 3: Embed
168 vectors (384 dimensions each)
     ↓  Step 4: Store
FAISS / Chroma Vector Database
     ↓  Step 5: Search
Ask question → Get relevant paragraph ✅
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| PDF Loading | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (HuggingFace) |
| Vector DB 1 | FAISS (Facebook AI Similarity Search) |
| Vector DB 2 | Chroma (alternative) |

---

## 📦 Setup

### 1. Clone the repository
```bash
git clone https://github.com/mahivbhosale/rag-pipeline-hexaware.git
cd rag-pipeline-hexawar
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your PDF
Place your PDF file in the project folder and update `rag_pipeline.py`:
```python
PDF_FILE_PATH = "your_document.pdf"  # Line 24
```

---

## 🚀 How to Run

### Run the complete pipeline:
```bash
python rag_pipeline.py
```

### Run the test suite:
```bash
python test_rag_pipeline.py
```

---

## ⚙️ Configuration

All settings are at the top of `rag_pipeline.py`:

```python
PDF_FILE_PATH = "ml_notes.pdf"                              # Your PDF
CHUNK_SIZE = 500                                            # Characters per chunk
CHUNK_OVERLAP = 50                                          # Overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # HuggingFace model
DB_TYPE = "faiss"                                           # "faiss" or "chroma"
DB_FOLDER = "./vector_db"                                   # Save location
```

---

## 📊 Sample Results (ml_notes.pdf)

```
Pages loaded    : 42
Chunks created  : 168
Embedding dim   : 384
Vectors stored  : 168
DB saved to     : ./vector_db/faiss_index
```

---

## 📁 Project Structure

```
rag-pipeline-hexawar/
├── rag_pipeline.py         ← Main pipeline code
├── test_rag_pipeline.py    ← Complete test suite
├── requirements.txt        ← All dependencies
├── README.md               ← This file
├── QUICKSTART.md           ← 5-minute setup guide
├── RAG_GUIDE.md            ← Detailed documentation
├── COMPLETE_SOLUTION.md    ← Full walkthrough
└── ml_notes.pdf            ← Sample input PDF
```

---

## 🔍 How Search Works

```python
from rag_pipeline import load_vector_db, search_vector_db, create_embeddings

# Load saved database
embeddings = create_embeddings()
vector_db = load_vector_db(embeddings=embeddings)

# Search by meaning (semantic search)
results = search_vector_db(vector_db, "What is gradient descent?", k=3)
```

---

## 📚 Key Concepts

| Concept | Explanation |
|---|---|
| **RAG** | Connects AI to your own documents for grounded answers |
| **Chunking** | Splits large text into searchable pieces |
| **Embeddings** | Converts text to numbers that capture meaning |
| **Vector DB** | Stores and searches embeddings by similarity |
| **Semantic Search** | Finds content by meaning, not just keywords |

---

## ⚠️ Notes

- First run downloads embedding model (~90 MB) — cached after that
- `vector_db/` folder is generated automatically — not committed to git
- Both FAISS and Chroma implementations are included
