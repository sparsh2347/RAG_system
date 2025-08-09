# main.py
import os
import json
import hashlib
from dotenv import load_dotenv
from services.extraction import extract_text_from_file
from services.chunker import HybridChunker
from services.embedder import Embedder
from services.storage import VectorStore
from services.retriever import Retriever
from services.evaluator import Evaluator

if os.path.exists(".env"):
    load_dotenv()

PROCESSOR_ID=os.getenv("PROCESSOR_ID")
PROJECT_ID=os.getenv("PROJECT_ID")

INDEX_PATH = "data/faiss_index.bin"
METADATA_PATH = "data/metadata.json"
CACHE_FILE = "data/processed_docs.json"
from google.cloud import storage
from io import BytesIO

BUCKET_NAME = os.getenv("BUCKET_NAME")
storage_client = storage.Client()

def _load_cache():
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("processed_docs.json")
    if blob.exists():
        data = blob.download_as_bytes().decode("utf-8")
        return json.loads(data)
    return {}

def _save_cache(cache):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("processed_docs.json")
    blob.upload_from_string(json.dumps(cache))
    print("✅ Cache saved to GCS.")


def _hash_doc_id(doc_path: str) -> str:
    return hashlib.sha256(doc_path.encode()).hexdigest()


def process_document(file_path: str):
    """
    Step 1-4: Extract text → Chunk → Embed → Store in vector DB
    """
    print(f"\n📂 Processing document: {file_path}")

     # Load cache
    cache = _load_cache()
    doc_hash = _hash_doc_id(file_path)

    # Skip if already processed
    if doc_hash in cache:
        print("⚡ Document already processed. Skipping reprocessing.")
        return
    
    # Extract text from PDF using Cloud Document AI
    text = extract_text_from_file(file_path,PROCESSOR_ID,PROJECT_ID)

    # Chunk the extracted text
    chunker = HybridChunker()
    raw_chunks = chunker.chunk(text)
    # chunks = [{"content": chunk.text, "metadata": chunk.metadata, "id":chunk.id, "index":chunk.index} for chunk in raw_chunks]

    # Embed chunks
    embedder = Embedder()
    embeddings = embedder.embed_chunks(raw_chunks)

    # Store embeddings + metadata
    store = VectorStore(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    store.add_embeddings(embeddings)
    print("Embeddings Added!!")
    store.save()

    # Save this document in cache
    cache[doc_hash] = file_path
    _save_cache(cache)

    print("✅ Document processed and stored successfully.")


def query_system(query: str, top_k: int = 5):
    """
    Step 5-6: Retrieve relevant chunks → Generate final answer
    """
    print(f"\n🔍 User Query: {query}")

    # Retrieve relevant chunks
    retriever = Retriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)

    # Evaluate & generate final answer
    evaluator = Evaluator()
    answer = evaluator.generate_answer(query, retrieved_chunks)

    print("\n🤖 Final Answer:\n")
    print(answer)
    return answer


if __name__ == "__main__":
    # === Example flow ===
    # Step 1: Process documents (only once per document)
    process_document("sample_files/sample.pdf")
    
    # Step 2: Query the system
    query_system("Does this policy cover knee surgery?", top_k=3)