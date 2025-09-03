from fastapi import FastAPI, UploadFile, File
from extract_text import extract_text_from_pdf, extract_text_from_image
from summarize import summarize_text, chunk_text
import os

app = FastAPI(title="Legal Document Summarizer")

@app.post("/summarize")
async def summarize_document(file: UploadFile = File(...)):
    filename = file.filename
    content = await file.read()
    temp_path = f"temp_{filename}"

    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(temp_path)
        elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(temp_path)
        else:
            return {"error": "Unsupported file type. Use PDF or image."}

        chunks = chunk_text(text, chunk_size=1000)
        summaries = [summarize_text(chunk) for chunk in chunks]
        final_summary = "\n".join(summaries)

    finally:
        os.remove(temp_path)

    return {"filename": filename, "summary": final_summary}
