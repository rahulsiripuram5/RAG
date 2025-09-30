import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
INDEX_DIR = "faiss_index_github" # The directory of your new index
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- SCRIPT ---

# 1. Load the saved index, chunks, and the model
print("Loading index and model...")
index = faiss.read_index(os.path.join(INDEX_DIR, "github_issues.index"))
with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)
print("Loaded successfully.")

# 2. Define a query and search
query = "How to fix high CPU usage on MacOS?"
k = 5  # Retrieve the top 5 most relevant chunks

print(f"\nSearching for: '{query}'")
query_embedding = model.encode([query], convert_to_tensor=False)
distances, indices = index.search(query_embedding.astype('float32'), k)

# 3. Print the results
print(f"\nTop {k} relevant chunks:\n")
for i in indices[0]:
    print("--- CHUNK ---")
    print(chunks[i])
    print("-------------\n")