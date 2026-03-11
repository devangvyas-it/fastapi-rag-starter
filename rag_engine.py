# Copyright 2026 [Devang Vyas]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GenerationConfig
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

# uncomment below to check available tasks
# from transformers.pipelines import PIPELINE_REGISTRY
# print(PIPELINE_REGISTRY.get_supported_tasks())

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1
        )

# Global storage
text_chunks = []
index = None

EMBEDDING_DIR = "data/embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

index_path = f"{EMBEDDING_DIR}/faiss_index.bin"
chunks_path = f"{EMBEDDING_DIR}/chunks.pkl"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)

if os.path.exists(chunks_path):
    with open(chunks_path, "rb") as f:
        text_chunks = pickle.load(f)

# Chunk size (characters)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)


def process_document(file_path):
    global index
    global text_chunks

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text
    chunks = chunk_text(text)

    text_chunks.extend(chunks)

    # Generate embeddings
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    # Initialize FAISS index if needed
    if index is None:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, f"{EMBEDDING_DIR}/faiss_index.bin")
    with open(f"{EMBEDDING_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)


def ask_question(question):
    global index

    if index is None or index.ntotal == 0:
        return "No document has been processed yet.", 0.0

    # Encode question
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    # Search vector DB
    k = 3
    # Ensure we don't ask for more results than exist
    search_k = min(k, index.ntotal)
    distances, indices = index.search(question_embedding, search_k)

    if len(indices[0]) == 0:
        return "Could not find any relevant chunks.", 0.0

    retrieved_chunks = [text_chunks[i] for i in indices[0]]

    # The first result is the best match (smallest L2 distance)    
    best_distance = distances[0][0]

    # Convert L2 distance to cosine similarity.
    # The sentence-transformer model `all-MiniLM-L6-v2` produces normalized embeddings.
    # For normalized vectors, cos_sim = 1 - (L2_dist^2 / 2).
    similarity_score = 1 - (best_distance ** 2) / 2
    similarity_score = float(max(0.0, similarity_score))  # Clamp to 0+

    gen_config = GenerationConfig(
        max_new_tokens=50,
        pad_token_id=qa_pipeline.tokenizer.eos_token_id,  # Best practice for clean output
        return_full_text=False
    )

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    Answer the question using the context.
    Context: {context}
    Question: {question}
    """

    result = qa_pipeline(prompt, generation_config=gen_config)
    answer = result[0]["generated_text"]

    return answer, similarity_score