# RAG Chatbot for GitHub Issue Analysis

This project is an end-to-end Retrieval-Augmented Generation (RAG) system built to answer natural language questions about the GitHub issues of a major open-source repository (`microsoft/vscode`).

---

## Key Features
- **Data Ingestion**: Fetches and processes conversational data directly from the GitHub API.
- **Dense Retrieval**: Uses a sentence-transformer model (`multi-qa-MiniLM-L6-cos-v1`) to generate embeddings and **FAISS** for efficient similarity search.
- **Generative Answering**: Leverages a **FLAN-T5** model to synthesize coherent answers based on the retrieved context.
- **Performance Evaluation**: Includes a full evaluation pipeline using F1 and Exact Match scores to measure performance.

---

## Tech Stack
- **Python**
- **Hugging Face Transformers**
- **Sentence-Transformers**
- **FAISS**
- **PyGithub**

---

## Project Outcome
The system was successfully able to answer specific questions about the GitHub issues. It was evaluated on a manually created dataset, achieving a final **F1 Score of 0.2932**. The project involved significant data cleaning and prompt engineering to handle the noisy, unstructured nature of the source data.

---



## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/RAG.git](https://github.com/YOUR_USERNAME/RAG.git)
    cd RAG
    ```
2.  **Set up the environment:**
    ```bash
    conda create --name rag_env python=3.11 -y
    conda activate rag_env
    pip install -r requirements.txt
    ```
3.  **Run the pipeline:**
    - Add your GitHub token to `get_github_data.py`.
    - `python get_github_data.py` to fetch the data.
    - `python create_index.py` to build the search index.
    - `python ask_rag.py` to ask a single question.
    - `python evaluate.py` to run the full evaluation.
