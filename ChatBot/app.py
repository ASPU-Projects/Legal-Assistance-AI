from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import FastAPI

# إعدادات المسارات
DATA_DIR = "data"
DB_DIR = "chroma_db_nomic"

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
print("📂 Loading existing Chroma DB...")
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

# إعداد FastAPI
app = FastAPI(title="Legal Chatbot API", description="استعلامات قانونية باستخدام Chroma + Nomic embeddings")

# تعريف موديل الإدخال
class QueryRequest(BaseModel):
    question: str
    k: int = 1

@app.post("/ask")
def ask(request: QueryRequest):
    results = db.similarity_search(request.question, k=request.k)
    if results:  
        answer = results[0].page_content   # أخذ أول نتيجة فقط
    else:
        answer = "لم يتم العثور على نتائج."
    return {
        "query": request.question,
        "answer": answer
    }
