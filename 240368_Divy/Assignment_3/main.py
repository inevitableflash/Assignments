from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import tempfile
import os
from pypdf import PdfReader

# Import our custom chunking function
from chunk import split_text_into_chunks

app = FastAPI(title="Text Chunking API")

# --- Pydantic Models for Input Validation ---
class TextChunkingRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The large text to be chunked")
    chunk_size: int = Field(default=50, gt=0, description="Size of each chunk")
    chunk_overlap: int = Field(default=10, ge=0, description="Overlap between chunks")

class ChunkingResponse(BaseModel):
    total_chunks: int
    chunks: list[str]

# --- 1. Welcome Endpoint ---
@app.get("/")
def welcome():
    return {"message": "Welcome to the AI Text Chunking API!"}

# --- 2. Text Chunking Endpoint ---
@app.post("/chunk", response_model=ChunkingResponse)
def chunk_text(request: TextChunkingRequest):
    try:
        # Call the function from chunk.py
        chunks, total = split_text_into_chunks(
            text=request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        return {"total_chunks": total, "chunks": chunks}
    except Exception as e:
        # Error handling
        raise HTTPException(status_code=500, detail=f"An error occurred during chunking: {str(e)}")

# --- 3. EXTRA FEATURE: PDF Upload & Text Extraction ---
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
            
        # Extract text from PDF
        reader = PdfReader(temp_file_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
            
        # Clean up the temporary file
        os.unlink(temp_file_path) 
        
        return {"filename": file.filename, "extracted_text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")