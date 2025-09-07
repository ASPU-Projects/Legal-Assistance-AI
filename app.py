from fastapi import FastAPI, UploadFile, File
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from Recommedation_System.Preprocessing import Preprocess
import os
# from Document_Analysis.extract_text import extract_text_from_pdf, extract_text_from_image
# from Document_Analysis.summarize import summarize_text, chunk_text
from Recommedation_System.main import recommendation
 

app = FastAPI(title="Legal Assistance AI",
              description="""
A Three AI Systems work for this platform 
""",version="1.0.1")


# root 
@app.get("/")
def ready():
    return "Done"

# ChatBot


# إعدادات المسارات
DATA_DIR = "data"
DB_DIR = "ChatBot/chroma_db_nomic"

# تحميل نموذج Nomic
nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", 
    trust_remote_code=True
)

# كلاس لتغليف الـ embeddings
class NomicEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding_model = NomicEmbeddings(nomic_model)

# تحميل قاعدة البيانات Chroma
print("Loading existing Chroma DB...")
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

# تعريف موديل الإدخال
class QueryRequest(BaseModel):
    question: str
    k: int = 3

@app.post("/ask") 
def ask(request: QueryRequest):
     results = db.similarity_search(request.question, k=request.k) 
     answers = [r.page_content for r in results] 
     return { "query": request.question, "results": answers }


# Recommendation System
class UserRequest(BaseModel):
    user_text: str
@app.post("/get_recommanded")
def run_get_recommanded(request:UserRequest):
    return recommendation(request.user_text)



# Document Analysis 

@app.post("/summarize")
async def summarize_document(file: UploadFile = File(...)):
    filename = file.filename
    content = await file.read()
    temp_path = f"temp_{filename}"

    # with open(temp_path, "wb") as f:
    #     f.write(content)

    # try:
    #     if filename.lower().endswith(".pdf"):
    #         text = extract_text_from_pdf(temp_path)
    #     elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
    #         text = extract_text_from_image(temp_path)
    #     else:
    #         return {"error": "Unsupported file type. Use PDF or image."}

    #     chunks = chunk_text(text, chunk_size=1000)
    #     summaries = [summarize_text(chunk) for chunk in chunks]
    #     final_summary = "\n".join(summaries)

    # finally:
    #     os.remove(temp_path)

    return {"filename": filename, "summary": "final_summary"}