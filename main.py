from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractedTextResponse(BaseModel):
    summary: str

# OCR and NLP function
def extract_text_and_summarize(image: bytes) -> str:
    try:
        # Convert image bytes to a PIL image
        image = Image.open(io.BytesIO(image))

        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(image)

        # Process with NLP
        doc = nlp(extracted_text)
        sentences = [sent.text for sent in doc.sents]

        # Generate a simple summary (first 3 sentences)
        summary = " ".join(sentences[:3]) if len(sentences) > 3 else extracted_text

        return summary.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.post("/extract-text/", response_model=ExtractedTextResponse)
async def extract_text(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        summary = extract_text_and_summarize(image_bytes)
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}