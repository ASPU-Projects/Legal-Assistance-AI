from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

def chunk_text(text, chunk_size=500):
    sentences = re.split(r'[.؟!\n]', text)  # Arabic-friendly splitting
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def embed_documents(documents):
    model = SentenceTransformer("CAMeL-Lab/bert-base-arabic-camelbert-mix")
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks, model
