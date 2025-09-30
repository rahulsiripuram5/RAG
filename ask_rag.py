import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import re

# --- CONFIGURATION ---
INDEX_DIR = "faiss_index_github"
EMBEDDING_MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1'
# Upgrade back to the 'base' model now that our input is clean
GENERATOR_MODEL_NAME = 'google/flan-t5-base' 

def clean_text(text):
    """Removes markdown tables and extra whitespace."""
    text = re.sub(r'\|.*\|', '', text)
    text = re.sub(r'---', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 1. LOAD ALL COMPONENTS ---
print("Loading retriever components...")
index = faiss.read_index(os.path.join(INDEX_DIR, "github_issues.index"))
with open(os.path.join(INDEX_DIR, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Retriever loaded.")

print("Loading generator model...")
tokenizer = T5Tokenizer.from_pretrained(GENERATOR_MODEL_NAME)
generator_model = T5ForConditionalGeneration.from_pretrained(GENERATOR_MODEL_NAME)
print("Generator loaded.")

# --- 2. DEFINE THE RAG PIPELINE FUNCTION ---
def answer_question(query, k=5):
    print(f"\n1. Retrieving relevant documents for: '{query}'")
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    cleaned_chunks = [clean_text(chunk) for chunk in retrieved_chunks]
    context = "\n\n".join(cleaned_chunks)

    # FINAL "Chain-of-Thought" PROMPT
    prompt = f"""
    You are a helpful assistant. Your task is to answer a question based on the provided context from GitHub issues.

    First, think step-by-step to analyze the context and find the key information needed to answer the question.
    Second, based on your analysis, provide a concise, final answer.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    RESPONSE:
    """

    print("2. Generating answer...")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = generator_model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("3. Answer generated.")
    return answer.strip()

# --- 3. ASK A QUESTION ---
if __name__ == "__main__":
    my_query = "What are users saying about performance issues or high CPU usage on Mac?"
    final_answer = answer_question(my_query)

    print("\n=====================================")
    print(f"QUERY: {my_query}")
    print("\nGENERATED ANSWER:")
    print(final_answer)
    print("=====================================")
