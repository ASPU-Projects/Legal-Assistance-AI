import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer


# Variables for the system
DATA_DIR = "data"       
DB_DIR = "chroma_db_nomic"

# AI model
nomic_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# For using chroma_db in new version of sentenceTransformer
class NomicEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding_model = NomicEmbeddings(nomic_model)

# Read pdf files 
documents = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        docs = loader.load()
        for d in docs:
            text = d.page_content.strip()           
            if len(text) > 30 and not text.replace("\n","").isdigit():
                documents.append(d)

print(f"📂 Number of Files: {len(os.listdir(DATA_DIR))}")
print(f"📑 Pages with real content: {len(documents)}")

# Split text to build chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"🔹 Number of chunks: {len(chunks)}")
print("⚙️ Building new Chroma DB...")

# Create chroma_db
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_DIR
)
db.persist()
print("✅ New Chroma DB created")