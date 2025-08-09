import os
import json
import faiss
import numpy as np
import tempfile
from google.cloud import storage
from io import BytesIO
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

class VectorStore:
    def __init__(self, bucket_name=None, index_path="data/faiss_index.bin", metadata_path="data/metadata.json", dim=768):
        self.bucket_name = bucket_name or os.getenv("BUCKET_NAME")
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.storage_client = storage.Client()

    def _upload_to_gcs(self, file_bytes, destination_blob_name):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(file_bytes, rewind=True)
        print(f"✅ Uploaded {destination_blob_name} to GCS bucket {self.bucket_name}")

    def _download_from_gcs(self, source_blob_name):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def add_embeddings(self, embedded_chunks):
        vectors = [chunk["embedding"] for chunk in embedded_chunks]
        vectors = np.array(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(embedded_chunks)

    def save(self):
        # Create a temp file, close it so FAISS can write on Windows
        tmp_index_file = tempfile.NamedTemporaryFile(suffix=".faiss", delete=False)
        try:
            tmp_index_file.close()  # Release file lock

            # Write FAISS index to temp file
            faiss.write_index(self.index, tmp_index_file.name)

            # Read index file bytes for upload
            with open(tmp_index_file.name, "rb") as f:
                index_bytes = BytesIO(f.read())

            self._upload_to_gcs(index_bytes, self.index_path)
        finally:
            # Delete the temp file
            os.remove(tmp_index_file.name)

        # Save metadata as bytes and upload
        metadata_bytes = BytesIO(json.dumps(self.metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        self._upload_to_gcs(metadata_bytes, self.metadata_path)

    def load(self):
        index_data = self._download_from_gcs(self.index_path)
        metadata_data = self._download_from_gcs(self.metadata_path)

        if not index_data or not metadata_data:
            raise FileNotFoundError("❌ Index or metadata not found in GCS.")

        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp_index_file:
            tmp_index_file.write(index_data)
            tmp_index_file.flush()  # Ensure data is written
            self.index = faiss.read_index(tmp_index_file.name)

        self.metadata = json.loads(metadata_data.decode("utf-8"))
        print("✅ Vector store loaded successfully from GCS.")

    def search(self, query_embedding, top_k=5):
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append({
                    "chunk": self.metadata[idx]["text"],
                    "metadata": self.metadata[idx]["metadata"],
                    "distance": float(dist)
                })
        return results
