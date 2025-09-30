import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "data_github"  # Directory where your GitHub issue files are stored
INDEX_DIR = "faiss_index_github" # Directory to save the new index
CHUNK_SIZE = 1000 # The size of each text chunk
OVERLAP = 200 # Overlap between chunks
MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1' # Use the QA-focused model

# --- SCRIPT ---

def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 1. Load all documents from the data directory
print(f"Loading documents from '{DATA_DIR}'...")
all_texts = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
print(f"Loaded {len(all_texts)} documents.")

# 2. Split all documents into chunks
print("Splitting documents into chunks...")
all_chunks = []
for text in all_texts:
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    all_chunks.extend(chunks)
print(f"Created {len(all_chunks)} chunks.")

# 3. Load the embedding model
print(f"Loading sentence transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# 4. Create embeddings for all chunks
print("Creating embeddings for all chunks...")
chunk_embeddings = model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=True)
chunk_embeddings = np.array(chunk_embeddings).astype('float32')
print(f"Embeddings created with shape: {chunk_embeddings.shape}")

# 5. Build and save the FAISS index
print("Building and saving FAISS index...")
d = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)

os.makedirs(INDEX_DIR, exist_ok=True)
faiss.write_index(index, os.path.join(INDEX_DIR, "github_issues.index"))

with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index and chunks saved successfully.")