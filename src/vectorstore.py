"""
FAISS Vector Store
- Embeds chunks (SentenceTransformers)
- Vector search (FAISS)
- Duplicate guard for same source
"""
import os
import pickle
from typing import Dict, List, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss.index"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index_path = index_path
        self.metadata_path = index_path + ".meta"

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata: List[Dict] = pickle.load(f)
            print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            print("✅ Created new FAISS index")

    def add_documents(self, chunks: List[str], source: str):
        # Duplicate guard
        if any(m["source"] == source for m in self.metadata):
            print(f"⚠️ {source} already indexed, skipping")
            return

        if not chunks:
            print("⚠️ No chunks to add")
            return

        print(f"🔄 Embedding {len(chunks)} chunks from {source} ...")
        embeddings = self.model.encode(chunks, convert_to_numpy=True).astype("float32")
        self.index.add(embeddings)

        for i, text in enumerate(chunks):
            self.metadata.append({"text": text, "source": source, "chunk_id": i})

        self.save()
        print(f"✅ Added {len(chunks)} chunks from {source}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        qemb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(qemb, min(top_k, self.index.ntotal))
        out: List[Tuple[Dict, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadata):
                out.append((self.metadata[idx], float(dist)))
        return out

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def get_stats(self) -> Dict:
        sources = list({m["source"] for m in self.metadata})
        return {"total_chunks": self.index.ntotal, "total_documents": len(sources), "sources": sources}
