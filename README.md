# FastAPI RAG Starter

A lightweight, self-contained **Retrieval-Augmented Generation (RAG)** application built with **FastAPI**. This API allows users to upload text documents, automatically indexes them for semantic search, and answers questions based on the uploaded content using a locally running Large Language Model (LLM).

## Features

- **Document Ingestion**: Upload `.txt` files which are automatically chunked and processed.
- **Vector Search**: Uses **Qdrant** and **Sentence Transformers** (`all-MiniLM-L6-v2`) for efficient semantic retrieval.
- **Context-Aware QA**: Generates natural language answers using the **Google Flan-T5** model (`google/flan-t5-base`).

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **ML/NLP**: Hugging Face Transformers, Sentence-Transformers, PyTorch
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

4. **Configuration (Optional):**
   Create a `.env` file in the root directory if you need to provide a Hugging Face token:
   ```ini
   HF_TOKEN=your_hugging_face_token
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
