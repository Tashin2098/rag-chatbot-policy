"""
FAISS Vector Store
- Embeds chunks (Hugging Face Inference API — no local model, saves memory)
- Vector search (FAISS)
- Duplicate guard for same source
"""
import os
import pickle
import time
from typing import Dict, List, Tuple
import faiss
import numpy as np
import requests


class FAISSVectorStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "faiss.index",
    ):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

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

    def _embed(self, texts: List[str], retries: int = 3) -> np.ndarray:
        """Call HF Inference API to get embeddings for a list of texts."""
        for attempt in range(retries):
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            if response.status_code == 200:
                result = response.json()
                return np.array(result, dtype="float32")
            elif response.status_code == 503:
                # Model is loading on HF's side, wait and retry
                print(f"⏳ Model loading on HF, retrying ({attempt + 1}/{retries})...")
                time.sleep(5)
            else:
                raise RuntimeError(f"HF API error {response.status_code}: {response.text}")
        raise RuntimeError("HF API failed after retries")

    def add_documents(self, chunks: List[str], source: str):
        if any(m["source"] == source for m in self.metadata):
            print(f"⚠️ {source} already indexed, skipping")
            return
        if not chunks:
            print("⚠️ No chunks to add")
            return

        print(f"🔄 Embedding {len(chunks)} chunks from {source} ...")
        embeddings = self._embed(chunks)
        self.index.add(embeddings)

        for i, text in enumerate(chunks):
            self.metadata.append({"text": text, "source": source, "chunk_id": i})

        self.save()
        print(f"✅ Added {len(chunks)} chunks from {source}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []

        qemb = self._embed([query])
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
