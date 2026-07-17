from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from src.ingestion import DocumentIngestor
from src.vectorstore import FAISSVectorStore
from src.retrieval import Retriever
from src.generation import AnswerGenerator


app = FastAPI(title="RAG Policy Chatbot API", version="1.0.0")

# Allow the website (running on a different local port, e.g. Live Server's
# 127.0.0.1:5500) to call this API from the browser. "*" is fine for local
# testing — restrict this to your real domain once deployed.
origins = [
    "http://localhost:5500",                  # For local testing
    "http://127.0.0.1:5500",                 # For local testing
    "https://talent-bridge-global-rosy.vercel.app"  # Your production Vercel URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


vs = FAISSVectorStore()
retriever = Retriever(vs)
try:
    gen = AnswerGenerator()
except ValueError as e:
    gen = None
ingestor = DocumentIngestor()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return vs.get_stats()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF/TXT supported")
    path = os.path.join("data", file.filename)
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())
    chunks, name = ingestor.process_document(path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text")
    vs.add_documents(chunks, name)
    return {"status": "success", "filename": name, "chunks_created": len(chunks)}


@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    """
    Retrieve top-k relevant document chunks with full text.
    Returns: list of chunks with complete text + metadata + relevance scores.
    """
    if vs.get_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="Index is empty.")

    chunks, context, citations = retriever.retrieve_with_chunks(req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "top_chunks": chunks,  # Full text + metadata for each chunk
        "citations": citations,
        "formatted_context": context  # For generation endpoint
    }


@app.post("/generate")
async def generate(req: QueryRequest):
    """
    Generate a conversational answer about Hireline directly from the LLM,
    with no document retrieval required. The website's chat widget calls
    this endpoint — it just needs a query in, an answer out, no ingestion
    step needed first.
    """
    if gen is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    # No retrieval — pass an empty context straight to the LLM. The system
    # prompt in AnswerGenerator already handles empty context gracefully,
    # answering from its built-in knowledge of Hireline instead of refusing.
    answer = gen.generate_answer(req.query, context="")

    return {"query": req.query, "answer": answer}
