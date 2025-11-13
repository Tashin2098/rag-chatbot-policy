# 📚 Company Policy RAG Chatbot

> A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that ingests company policy documents and answers employee questions with **grounded, cited responses—no hallucinations**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-darkgreen)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [How It Works](#-how-it-works)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)


---

## Features

✅ **Multi-Document Support** — Ingest multiple PDF/TXT policy files simultaneously  
✅ **ChatGPT-style UI** — Streamlit interface with sidebar chat history and per-chat memory  
✅ **Per-Chat Isolation** — Each conversation has its own FAISS index and file uploads  
✅ **Semantic Search** — SentenceTransformers embeddings + FAISS vector indexing  
✅ **Grounded Responses** — LLM answers strictly from retrieved context (no hallucination)  
✅ **Source Citations** — Every answer includes chunk references and relevance scores  
✅ **REST API** — FastAPI endpoints for ingest, retrieve, and generate  
✅ **Greeting Support** — Bot responds to "hi/hello" even before documents are uploaded  
✅ **Retrieved Context Transparency** — View exact chunks used for answer generation  

---

## Architecture

### System Flow Diagram

```
User Input (Streamlit/API)
    ↓
Document Processing → Chunking → Metadata Tagging
    ↓
Embedding (SentenceTransformers) → FAISS Indexing (Per-Chat)
    ↓
User Query
    ↓
Vector Search (Semantic Match) → Top-k Retrieval
    ↓
Context Formatting + Citations
    ↓
Groq LLM Generation (Grounded Prompt)
    ↓
Response with [Source X] Citations
    ↓
UI Display (Answer + Sources + Context)
```

### Module Architecture

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **src/ingestion.py** | Document processing & chunking | `process_document()` - Parse PDF/TXT, chunk, preserve metadata |
| **src/vectorstore.py** | FAISS index management | `add_documents()`, `search()`, `get_stats()` |
| **src/retrieval.py** | Semantic search & citations | `retrieve()`, `retrieve_with_chunks()` |
| **src/generation.py** | LLM integration | `generate_answer()` - Groq with grounded prompt |
| **app.py** | Streamlit UI | ChatGPT-style multi-chat interface |
| **api.py** | FastAPI REST endpoints | `/ingest`, `/retrieve`, `/generate`, `/stats` |

---

## Quick Start

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Groq API Key** (free at [console.groq.com](https://console.groq.com))
- **2GB RAM** minimum (for FAISS + embeddings)

### Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/Tashin2098/rag-chatbot-policy.git
cd rag-chatbot-policy
```

#### Step 2: Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# On Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Configure Environment

Create `.env` file in project root:

```bash
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
EOF
```

**Get API key:**
1. Visit https://console.groq.com
2. Sign up (free tier available)
3. Generate API key
4. Paste into `.env`

#### Step 5: Run Application

**Option A: Streamlit UI (Recommended)**

```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

**Option B: FastAPI REST API**

```bash
uvicorn api:app --reload --port 8000
```

Swagger UI at: `http://localhost:8000/docs`

---

## Usage Guide

### Streamlit UI Tutorial

#### 1. Start Application

```bash
streamlit run app.py
```

#### 2. Create New Chat

- Click **"+ New chat"** button
- Chat appears in sidebar as "Chat 1", "Chat 2", etc.

#### 3. Upload Policy Documents

- Click **"Upload files to this chat"** expander
- Select PDF/TXT files (supports multiple)
- Click **"Process files"**
- Wait for indexing notification

#### 4. Ask Questions

Type your question and press Enter:

```
"What is the annual leave policy?"
"How do I request remote work?"
"What are the probation terms?"
```

#### 5. View Results

- **Answer** — Main response text
- **Sources** — Expand to see file names + chunk IDs + relevance scores
- **Retrieved context** — Expand to see exact policy text used

#### 6. Switch Chats

Select different chat from sidebar dropdown. Each maintains separate memory and documents.

### Example Queries & Expected Outputs

| Query | Expected Output |
|-------|-----------------|
| "What is the sick leave policy?" | Sick leave days, documentation, process + [Source X] |
| "How do I request remote work?" | Remote work conditions, approval process + citations |
| "Stock option policy?" | "I cannot find this information in the provided documents" |
| "What's the probation period?" | Duration, terms, evaluation criteria + sources |

---

## How It Works

### Phase 1: Document Ingestion

```python
# PDF Upload → Text Extraction → Chunking → Metadata
1. Extract text from PDF/TXT (pypdf library)
2. Split into chunks (500 characters with 100 char overlap)
3. Attach metadata: {text, source_file, chunk_id}
4. Store for embedding
```

**Why overlap?** Preserves context across chunk boundaries—clauses won't be cut off.

### Phase 2: Embedding & Indexing

```python
# Text → Vectors → FAISS Index
1. Embed each chunk using SentenceTransformers (all-MiniLM-L6-v2)
   - 384-dimensional vectors
   - Fast (5ms per chunk)
2. Insert into per-chat FAISS index
3. Persist to disk: faiss_<chat_id>.index
4. Duplicate guard: skip re-indexing identical chunks
```

**Why all-MiniLM-L6-v2?**
- Fast (CPU compatible)
- Lightweight (22MB model)
- Excellent semantic understanding for policy text

### Phase 3: Semantic Search

```python
# Query → Vector Search → Top-k Chunks
1. Embed user query (same encoder as chunks)
2. FAISS finds nearest neighbors (cosine similarity)
3. Retrieve top-k=3 chunks
4. Calculate relevance: 1/(1+distance)
5. Format: [Source X: filename]\n{chunk_text}
```

### Phase 4: Grounded Answer Generation

```python
# Context + Query → LLM → Grounded Answer
System Prompt:
  "Answer ONLY from provided context.
   If not found, say: 'I cannot find this information'.
   Cite with [Source X]. Be concise and accurate."

Config:
  - Model: llama-3.3-70b-versatile
  - Temperature: 0.2 (consistency over creativity)
  - Max tokens: 500
```

**Why temperature=0.2?** Reduces hallucination, ensures focus on retrieved facts.

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### GET /

Health check

```bash
curl http://localhost:8000/
```

**Response:** `{"status": "ok"}`

#### GET /stats

Get index statistics

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "total_chunks": 42,
  "index_size_mb": 2.5
}
```

#### POST /ingest

Upload and index document

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@data/policy.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "policy.pdf",
  "chunks_created": 42
}
```

#### POST /retrieve

Retrieve top-k relevant chunks

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the sick leave policy?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "query": "What is the sick leave policy?",
  "top_chunks": [
    {
      "text": "Employees are entitled to 14 days of sick leave per calendar year...",
      "source": "Human_Resource_Policy.pdf",
      "chunk_id": 30,
      "relevance_score": 0.57
    },
    ...
  ],
  "citations": ["Human_Resource_Policy.pdf (Chunk 30, Relevance: 0.57)", ...],
  "formatted_context": "[Full context string]"
}
```

#### POST /generate

Generate grounded answer

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the sick leave policy?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "query": "What is the sick leave policy?",
  "answer": "According to the policy, employees are entitled to 14 days of sick leave per calendar year [Source 1]...",
  "sources": [
    "Human_Resource_Policy.pdf (Chunk 30, Relevance: 0.57)",
    "Human_Resource_Policy.pdf (Chunk 29, Relevance: 0.51)",
    "Human_Resource_Policy.pdf (Chunk 31, Relevance: 0.49)"
  ]
}
```

### Using Swagger UI

Interactive API testing at: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
rag-chatbot-policy/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── ingestion.py             # Document processing & chunking
│   ├── vectorstore.py           # FAISS index management
│   ├── retrieval.py             # Semantic search & citations
│   └── generation.py            # Groq LLM integration
│
├── app.py                       # Streamlit UI (ChatGPT-style)
├── api.py                       # FastAPI REST endpoints
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git exclusions
└── README.md                    # This file

# Auto-generated (not in repo, see .gitignore)
├── data/                        # User-uploaded PDFs
├── faiss_*.index                # Per-chat FAISS indexes
└── faiss_*.index.meta           # Index metadata
```

---

## Testing

### Test Query 1: Basic Retrieval

```
Query: "What is the annual leave entitlement?"
✅ Expected: Returns 3 HR policy chunks with relevance > 0.5
🔍 Check: Verify via /retrieve endpoint
```

### Test Query 2: Generation Quality

```
Query: "How many days of leave do employees get?"
✅ Expected: Answer with specific numbers + [Source X] citations
🔍 Check: Relevant chunks support the answer?
```

### Test Query 3: No-Hallucination

```
Query: "What is the company's cryptocurrency policy?"
✅ Expected: "I cannot find this information in the provided documents"
🔍 Check: Bot doesn't invent plausible-sounding false policies
```

### Test Query 4: Multi-Chat Isolation

```
Chat 1: Upload HR_Policy.pdf
Chat 2: Upload IT_Security.pdf
Query in Chat 1: "Password policy?"
✅ Expected: Returns "not found" (Chat 1 has only HR docs)
🔍 Check: No cross-chat data leakage
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not found` | Create `.env` file with your API key |
| FAISS index corrupted | Delete `.index` and `.index.meta` files; re-upload |
| Slow query response | Reduce `chunk_size` or switch to GPU FAISS |
| Out of memory | Reduce `top_k` or use smaller embedding model |
| Hallucinated answers | Increase system prompt enforcement |
| Streamlit port already in use | `streamlit run app.py --server.port 8502` |
| FastAPI port conflict | Change port in `api.py` (`port=8001`) |

---

## No-Hallucination Strategy

**Multi-layered Approach:**

1. **System Prompt** — "Answer ONLY from provided context"
2. **Low Temperature** — `0.2` (deterministic, fact-focused)
3. **Context Transparency** — UI shows exact chunks used
4. **Citation Requirement** — Every claim must cite [Source X]
5. **Explicit Refusal** — "I cannot find..." when needed

**Test:** Ask about non-existent policies
- ✅ Bot: "I cannot find this information in the provided documents"
- ❌ Never: Fabricates plausible-sounding policies

---

## Performance

| Metric | Value |
|--------|-------|
| **Document Ingestion** | ~100ms per page |
| **Query Embedding** | ~5ms |
| **FAISS Search** | ~10ms (top-k=3) |
| **LLM Generation** | ~2-3 seconds |
| **Total Response Time** | ~3-4 seconds |
| **Embedding Model Size** | 22MB |
| **Index Size** | ~100KB per 100 chunks |

---

## Advanced Configuration

### Enable GPU FAISS

```bash
# Install GPU version
pip install faiss-gpu

# Update code (auto-detected)
```



### Add User Authentication


## Learning Resources

- [RAG Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API Docs](https://console.groq.com/docs)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---


