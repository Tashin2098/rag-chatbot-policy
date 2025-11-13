from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from src.ingestion import DocumentIngestor
from src.vectorstore import FAISSVectorStore
from src.retrieval import Retriever
from src.generation import AnswerGenerator


app = FastAPI(title="RAG Policy Chatbot API", version="1.0.0")


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
    Generate grounded answer from retrieved context.
    """
    if vs.get_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="Index is empty.")
    if gen is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")
    
    # Use original retrieve method (no changes needed for UI compatibility)
    context, citations = retriever.retrieve(req.query, top_k=req.top_k)
    answer = gen.generate_answer(req.query, context)
    
    return {"query": req.query, "answer": answer, "sources": citations}