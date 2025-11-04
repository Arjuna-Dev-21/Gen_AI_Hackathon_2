# AI-Powered Document Search and Summarization System

This project is an AI-driven tool built with Streamlit and Hugging Face Transformers. It ingests documents (PDF, DOCX, TXT), performs semantic search, and generates concise summaries of the most relevant sections based on a user's query.

## Features
- **Document Ingestion:** Upload PDF, DOCX, and TXT files.
- **Text Processing:** Extracts, cleans, and splits text into manageable chunks.
- **Semantic Search:** Uses sentence-transformer embeddings and a FAISS vector store to find the most contextually relevant information.
- **AI Summarization:** Leverages the Llama-3.2-1B-Instruct model to synthesize the retrieved information into a coherent answer.

## How to Run
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac/Linux).
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your `HUGGINGFACE_HUB_TOKEN`.
6. Run the Streamlit app: `streamlit run app.py`

## Modules Used
- **Frontend:** Streamlit
- **Text Extraction:** PyPDF2, python-docx
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (CPU)
- **Summarization:** Hugging Face Transformers (`unsloth/Llama-3.2-1B-Instruct`)