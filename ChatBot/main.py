from PyPDF2 import PdfReader
import os

def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Folder containing your 17 PDFs
pdf_folder = "ChatBot/legal_docs"
all_texts = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        all_texts.append(load_pdf_text(file_path))

# Combine all PDF texts into one big string
full_text = "\n".join(all_texts)

def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

documents = split_text(full_text)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode chunks
embeddings = model.encode(documents)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

from transformers import pipeline

# You can replace 'gpt2' with a better local model or even an Arabic model
generator = pipeline("text-generation", model="gpt2")

def answer_query(query):
    context = "\n".join(retrieve(query))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=200, do_sample=True)
    return result[0]["generated_text"].split("Answer:")[-1].strip()


while True:
    q = input("You: ")
    if q.lower() in ["quit", "exit"]:
        break
    print("Bot:", answer_query(q))
