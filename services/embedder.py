# services/embedder.py
import os
import google.generativeai as genai # type: ignore
import numpy as np # type: ignore
from dotenv import load_dotenv # type: ignore

if os.path.exists(".env"):
    load_dotenv()
# from utils.logger import logger
# from utils.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = "models/embedding-001" 
from services.storage import VectorStore
class Embedder:
    """
    Handles generating embeddings for text chunks using Google's Gemini Embedding API.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY not found in environment")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = GEMINI_EMBEDDING_MODEL

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding vector for a single text input.
        """
        try:
            response = genai.embed_content(
                model=self.model,
                content=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"❌ Embedding failed for text: {e}")
            return None

    def embed_chunks(self, chunks: list) -> list:
        """
        Adds embeddings to each chunk in the list.
        Each chunk is expected to be a dict with at least 'text'.
        """
        print("----------Embedding Started-------!!")
        embedded_chunks = []
        for idx, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk["text"])
            if embedding:
                chunk["embedding"] = np.array(embedding, dtype=np.float32).tolist()
                embedded_chunks.append(chunk)
            else:
                print(f"⚠️ Skipping chunk index {idx} due to embedding failure")
        print("--------Embedding Completed!!--------")
        # print(embedded_chunks)
        return embedded_chunks


# if __name__ == "__main__":
#     # Test with a sample chunk
#     test_chunks = [
#         {"content": "Artificial Intelligence is transforming industries.", "metadata": {"source": "sample.pdf"}}
#     ]
#     embedder = Embedder()
#     embedded_chunks = embedder.embed_chunks(test_chunks)
#     print(embedded_chunks)
#     dim = len(embedded_chunks[0]["embedding"])
#     store = VectorStore(dim=dim)
#     store.add(
#         [ch["embedding"] for ch in embedded_chunks],
#         [ch["metadata"] for ch in embedded_chunks]
#     )
#     store.save()
