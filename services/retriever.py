import os
import numpy as np
from typing import List
from services.embedder import Embedder
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from google.cloud import storage
import faiss
import json
import tempfile
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

genai.configure(api_key=GEMINI_API_KEY)


class Retriever:
    """
    Retrieves top relevant document chunks for a given query.
    """

    def __init__(self, index_path="faiss_index.bin", metadata_path="metadata.json", heading_boost=0.25, dim=768):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.heading_boost = heading_boost
        self.dim = dim
        self.embedder = Embedder()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.storage_client = storage.Client()

        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        # Load index and metadata from GCS
        self.load()

    def _download_from_gcs(self, blob_name):
        bucket = self.storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"❌ {blob_name} not found in bucket {BUCKET_NAME}")
        return blob.download_as_bytes()

    import tempfile

    def load(self):
        # Download FAISS index
        index_bytes = self._download_from_gcs(self.index_path)
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp_file:
            tmp_file.write(index_bytes)
            tmp_file.flush()
            self.index = faiss.read_index(tmp_file.name)

        # Download metadata
        metadata_bytes = self._download_from_gcs(self.metadata_path)
        self.metadata = json.loads(metadata_bytes.decode("utf-8"))

        print("✅ Retriever: FAISS index & metadata loaded from GCS.")

    def _heading_match_score(self, query: str, headings: List[str]) -> float:
        """
        Compute semantic similarity between query and each heading.
        Return max similarity (0 to 1).
        """
        if not headings:
            return 0.0
        query_emb = self.model.encode([query], convert_to_tensor=True)
        heading_embs = self.model.encode(headings, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, heading_embs)[0].cpu().numpy()
        return float(np.max(sims))

    def expand_query(self, original_query: str) -> str:
        """
        Expands user query with synonyms and related legal/insurance terminology
        using Gemini LLM before retrieval.
        """
        prompt = f"""
        You are a domain expert in insurance and legal documents.
        Expand the following query by adding synonyms, related terms, and 
        possible alternative phrasings that may appear in the documents.
        Keep it short, comma-separated, and relevant.

        Example:
        Input: "Does this policy cover knee surgery?"
        Output: "knee surgery, orthopedic surgery, joint operation, surgical treatment for knee injury, knee replacement"

        Input query:
        {original_query}
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            expanded = response.candidates[0].content.parts[0].text.strip()
            return expanded
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return original_query

    def retrieve(self, query: str, top_k: int = 5):
        """
        Given a user query, retrieve top_k relevant chunks.
        """
        expanded_query = self.expand_query(query)
        query_embedding = self.embedder.get_embedding(expanded_query)
        if query_embedding is None:
            raise ValueError("❌ Failed to generate embedding for the query")

        # Search in FAISS index
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "chunk": self.metadata[idx]["text"],
                    "metadata": self.metadata[idx]["metadata"],
                    "distance": float(dist)
                })

        # Boost by heading similarity
        boosted_results = []
        for r in results:
            base_score = -r["distance"]
            headings = r["metadata"].get("headings", [])
            heading_sim = self._heading_match_score(query, headings)
            boosted_score = base_score + (heading_sim * self.heading_boost)
            boosted_results.append({**r, "boosted_score": boosted_score})

        boosted_results.sort(key=lambda x: x["boosted_score"], reverse=True)
        return boosted_results[:top_k]
