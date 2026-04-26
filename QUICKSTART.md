# 🚀 Quick Start Guide (5 Minutes)

## Step 1: Copy the Files

You have 3 files:
- `rag_pipeline.py` - Main script
- `requirements.txt` - Dependencies
- `RAG_GUIDE.md` - Full documentation

Copy all to your project folder.

---

## Step 2: Install Dependencies (2 minutes)

```bash
# Open terminal in your project folder and run:
pip install -r requirements.txt

# Wait for installation to complete...
```

---

## Step 3: Prepare Your PDF

1. Get your PDF file (e.g., `my_document.pdf`)
2. Place it in the same folder as `rag_pipeline.py`
3. Remember the filename!

---

## Step 4: Update the Script

Open `rag_pipeline.py` and change this line (line ~21):

```python
# BEFORE:
PDF_FILE_PATH = "your_document.pdf"

# AFTER (example):
PDF_FILE_PATH = "my_document.pdf"
```

---

## Step 5: Run the Script (1 minute)

```bash
python rag_pipeline.py
```

**What happens:**
- Downloads embedding model (first time only, ~100-500 MB)
- Reads your PDF
- Creates chunks
- Builds vector database
- Saves it to `./vector_db/` folder

**Output:**
```
🚀 STARTING RAG PIPELINE
📄 Loading PDF from: my_document.pdf
✅ Loaded 10 pages from PDF
🔪 Chunking documents...
✅ Created 45 chunks
🤖 Loading embedding model...
✅ Embedding model loaded
📦 Creating FAISS vector database...
✅ FAISS database saved
✅ PIPELINE COMPLETE!
🔍 TESTING VECTOR DATABASE
🔍 Searching for: 'What is the main topic?'
✅ Found 2 similar results
...
```

---

## Step 6: Your Vector DB is Ready! ✅

The database is saved in: `./vector_db/`

You can now:
- Query it with `search_vector_db()`
- Load it later without re-processing
- Use it in your RAG application

---

## 💡 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'langchain'"
```bash
# Solution: Reinstall
pip install --upgrade -r requirements.txt
```

### Issue: "FileNotFoundError: my_document.pdf"
```bash
# Solution: Check the path
# Make sure:
# 1. PDF exists in the same folder as script
# 2. Filename matches exactly (case-sensitive on Linux/Mac)
# 3. Or use full path: "/home/user/Documents/my_document.pdf"
```

### Issue: "Out of memory" error
```python
# Solution: Reduce chunk size in rag_pipeline.py
CHUNK_SIZE = 256  # Changed from 512
```

### Issue: "Connection error" or "Download failed"
```bash
# Solution: Check internet connection, then retry
# Model downloads automatically on first run
```

---

## 🔍 Next Steps

After running successfully, you can:

1. **Search the database programmatically:**
   ```python
   from rag_pipeline import load_vector_db, search_vector_db, create_embeddings
   
   embeddings = create_embeddings()
   vector_db = load_vector_db(embeddings=embeddings)
   
   # Search
   results = search_vector_db(vector_db, "your question here", k=5)
   ```

2. **Try different embedding models:**
   Change `EMBEDDING_MODEL` in the script
   - `"sentence-transformers/all-MiniLM-L6-v2"` (fast, good)
   - `"sentence-transformers/all-mpnet-base-v2"` (better quality)

3. **Try Chroma instead of FAISS:**
   Change `DB_TYPE = "chroma"` in the script

4. **Connect with LLM for Q&A:**
   Use OpenAI/Claude API + vector DB for RAG system

---

## 📚 File Structure After Running

```
project_folder/
├── rag_pipeline.py          ← Main script
├── requirements.txt         ← Dependencies
├── RAG_GUIDE.md            ← Full documentation
├── my_document.pdf         ← Your PDF
└── vector_db/              ← Created by script
    ├── faiss_index/        ← FAISS database files
    │   ├── index.faiss
    │   └── index.pkl
    └── chroma_db/          ← Chroma database files (if used)
```

---

## ✅ Success Checklist

- [ ] Downloaded/copied the 3 files
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Have your PDF file ready
- [ ] Updated `PDF_FILE_PATH` in `rag_pipeline.py`
- [ ] Run the script (`python rag_pipeline.py`)
- [ ] Vector database created in `./vector_db/`
- [ ] Search tests passed!

---

## 🎉 You're Done!

Your RAG pipeline is complete and working!

**Next for HexaWar:**
- Present this to your mentor
- Show the vector DB creation process
- Demonstrate search functionality
- Explain the chunking strategy

---

**Need help?** Refer to `RAG_GUIDE.md` for detailed explanations!
