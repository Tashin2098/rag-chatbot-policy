from typing import List, Tuple, Dict
from src.vectorstore import FAISSVectorStore


class Retriever:
    def __init__(self, vectorstore: FAISSVectorStore):
        self.vectorstore = vectorstore


    def retrieve(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        Original method for Streamlit UI - returns formatted context string + citations.
        No changes here - keeps UI functionality intact.
        """
        results = self.vectorstore.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found.", []

        context_parts: List[str] = []
        citations: List[str] = []
        for i, (m, dist) in enumerate(results, 1):
            sim = 1 / (1 + dist)
            context_parts.append(f"[Source {i}: {m['source']}, Chunk {m['chunk_id']}]\n{m['text']}")
            citations.append(f"{m['source']} (Chunk {m['chunk_id']}, Relevance: {sim:.2f})")

        context = "\n\n---\n\n".join(context_parts)
        return context, citations


    def retrieve_with_chunks(self, query: str, top_k: int = 3) -> Tuple[List[Dict], str, List[str]]:
        """
        New method for API - returns full chunk objects with text + metadata.
        Used by /retrieve endpoint to show complete chunk details.
        """
        results = self.vectorstore.search(query, top_k=top_k)
        if not results:
            return [], "No relevant documents found.", []

        chunks = []
        context_parts: List[str] = []
        citations: List[str] = []
        
        for i, (m, dist) in enumerate(results, 1):
            sim = 1 / (1 + dist)
            
            # Full chunk object for API response
            chunks.append({
                "text": m["text"],
                "source": m["source"],
                "chunk_id": m["chunk_id"],
                "relevance_score": round(sim, 2)
            })
            
            # Formatted context for generation
            context_parts.append(f"[Source {i}: {m['source']}, Chunk {m['chunk_id']}]\n{m['text']}")
            citations.append(f"{m['source']} (Chunk {m['chunk_id']}, Relevance: {sim:.2f})")

        context = "\n\n---\n\n".join(context_parts)
        return chunks, context, citations