from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_loader import load_pdfs_from_folder
from embed_and_index import embed_documents
from query_engine import get_top_chunks

app = FastAPI()

# Load documents3 and build index
docs = load_pdfs_from_folder("legal_docs")
index, chunks, model = embed_documents(docs)

# Request schema
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        answers = get_top_chunks(request.question, model, index, chunks)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
