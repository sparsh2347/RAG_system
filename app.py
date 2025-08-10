# app.py
import os
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List
from dotenv import load_dotenv
from main import process_document, query_system  # your existing logic functions
import uvicorn 

load_dotenv()

API_KEY = os.getenv("API_KEY")  # your secret token for auth

app = FastAPI(
    title="HackRx Retrieval API",
    description="API for document question answering with retrieval & GPT",
    version="1.0"
)

class HackRxRequest(BaseModel):
    documents: str  # URL or local path of the PDF document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

def check_auth(authorization: str):
    # print(authorization)
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(request: HackRxRequest, authorization: str = Header(None, alias="Authorization")):
    # 1. Auth check
    check_auth(authorization)

    # 2. Process the document (download, extract, chunk, embed, store)
    try:
        process_document(request.documents)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Document processing failed: {str(e)}"})

    # 3. For each question, run retrieval + evaluation to get answer
    answers = []
    for question in request.questions:
        try:
            # query_system currently prints answer; modify it to return answer string instead
            answer = query_system(question, top_k=5)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Failed to answer question: {str(e)}")

    return {"answers": answers}

# Optional: Root health check endpoint
@app.get("/")
async def root():
    return {"status": "HackRx Retrieval API is up and running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT env var or default to 8000 locally
    uvicorn.run("app:app", host="0.0.0.0", port=port)