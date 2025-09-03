from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_DIR = "chroma_db_nomic"


nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", 
    trust_remote_code=True
)

class NomicEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding_model = NomicEmbeddings(nomic_model)

print("📂 Loading existing Chroma DB...")
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model
)

query = "ما هي عقوبة السرقة في القانون السوري؟"
results = db.similarity_search(query, k=1)

print("\n🔍 نتائج البحث:")
for i, r in enumerate(results, 1):
    print(f"{i}- {r.page_content[:500]}...\n")