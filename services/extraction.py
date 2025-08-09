import os
from google.cloud import documentai_v1 as documentai
from mimetypes import guess_type
import requests
import PyPDF2
import tempfile
from dotenv import load_dotenv  # type: ignore

# Load .env only if running locally (Cloud Run sets env vars directly)
if os.path.exists(".env"):
    load_dotenv()

def extract_text_with_pypdf2(file_path: str) -> str:
    """
    Fallback extraction using PyPDF2 to extract text from PDF.
    """
    print("⚠️ Falling back to PyPDF2 extraction...")
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_file(file_path: str, processor_id: str = None, project_id: str = None, location: str = "us") -> str:
    """
    Extracts text from a PDF or DOCX file using Google Document AI.
    """
    try:
    # Fallback to environment variables if not passed explicitly
        processor_id = processor_id or os.getenv("PROCESSOR_ID")
        project_id = project_id or os.getenv("PROJECT_ID")

        if not processor_id or not project_id:
            raise ValueError("Missing PROCESSOR_ID or PROJECT_ID. Set them as environment variables or pass them explicitly.")

        print("-------Starting Extraction-------")

        if file_path.startswith(("http://", "https://")):
            print("Downloading file from URL...")
            response = requests.get(file_path, stream=True)
            if response.status_code != 200:
                raise Exception("Failed to download file from URL")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                file_path = tmp.name
                print(f"Saved URL content to temporary file: {file_path}")

        client = documentai.DocumentProcessorServiceClient()

        with open(file_path, "rb") as file:
            file_content = file.read()

        mime_type, _ = guess_type(file_path)
        if mime_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise ValueError(f"Unsupported file type: {mime_type}")

        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document,imageless_mode=True)

        result = client.process_document(request=request)
        print("--------Extraction Completed-------")
        return result.document.text
    except Exception as e:
        print(f"Google Document AI extraction failed: {e}")
        if mime_type == "application/pdf":
            result=extract_text_with_pypdf2(file_path)
            print("--------Extraction Completed-------")
            return result
        else:
            raise RuntimeError("No fallback available for this file type.")

