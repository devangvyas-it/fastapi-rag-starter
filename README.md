# FastAPI RAG Starter

A lightweight, self-contained **Retrieval-Augmented Generation (RAG)** application built with **FastAPI**. This API allows users to upload text documents, automatically indexes them for semantic search, and answers questions based on the uploaded content using either a high-performance cloud LLM (Groq) or a locally running model.

## Features

- **Document Ingestion**: Upload `.txt` files which are automatically chunked and processed.
- **Vector Search**: Uses **Qdrant** and **Sentence Transformers** (`all-MiniLM-L6-v2`) for efficient semantic retrieval.
- **Context-Aware QA**: Generates natural language answers using **Groq** (`llama-3.3-70b-versatile`) for speed and quality, falling back to **Google Flan-T5-Base** (`google/flan-t5-base`) for local execution.

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **ML/NLP**: Hugging Face Transformers, Sentence-Transformers, PyTorch, Groq API
- **Vector Store**: Qdrant

## Installation

### Prerequisites
- Python 3.10+

### Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**
   Create a `.env` file in the root directory.
   - `HF_TOKEN` (Optional): For accessing gated Hugging Face models.
   - `GROQ_API_KEY` (Optional): If provided, the system uses Groq's Llama 3 model. If missing, it defaults to the local Flan-T5 model.
   - `CHUNK_SIZE` (Optional): Character size for text splitting (default: 300).
   - `CHUNK_OVERLAP` (Optional): Character overlap for text splitting (default: 50).
   - `SIMILARITY_THRESHOLD` (Optional): Cosine similarity threshold (0.0-1.0) for filtering relevant chunks (default: 0.5).
   - `RETRIEVAL_LIMIT` (Optional): Number of chunks to retrieve (default: 3).

   ```ini
   HF_TOKEN=your_hugging_face_token
   GROQ_API_KEY=your_groq_api_key
   CHUNK_SIZE=300
   CHUNK_OVERLAP=50
   SIMILARITY_THRESHOLD=0.5
   RETRIEVAL_LIMIT=3
   ```

## Usage

### Running Locally
Start the server:
```bash
uvicorn app:app --reload
```
The API will be accessible at `http://localhost:8000`.

## API Endpoints

- **`GET /`**: Health check.
  ```bash
  curl -X GET "http://localhost:8000/"
  ```
- **`POST /upload`**: Upload a text document.
  ```bash
  curl -X POST "http://localhost:8000/upload" -F "file=@\"D:\docker.txt\""
  curl -X POST "http://localhost:8000/upload" -F "file=@\"D:\huggingface.txt\""
  ```
- **`POST /ask`**: Ask a question based on the uploaded documents.
  ```bash
  curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What are advantages of using docker?\"}"
  ```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
