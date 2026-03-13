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
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GenerationConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq

# uncomment below to check available tasks
# from transformers.pipelines import PIPELINE_REGISTRY
# print(PIPELINE_REGISTRY.get_supported_tasks())

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1
        )

# Configurable parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", 3))

# vector db client
db_client = None

EMBEDDING_DIR = "data/embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)
collection_name = "docs"

def get_db_client():
    global db_client
    if db_client is None:
        db_client = QdrantClient(path=EMBEDDING_DIR)
        collections = db_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            db_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
    return db_client

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
    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text
    chunks = chunk_text(text)
    # Generate embeddings
    embeddings = model.encode(chunks, normalize_embeddings=True)       

    embedding_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
       embedding_data.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
        )

    client = get_db_client()
    client.upsert(
        collection_name=collection_name,
        points=embedding_data
    )  


def ask_question(question):   

    # Encode question
    question_embedding = model.encode(question, normalize_embeddings=True)

    # Search vector DB
    # Ensure we don't ask for more results than exist    
    client = get_db_client()
    search_results = client.query_points(collection_name=collection_name,
                                        query=question_embedding.tolist(),
                                        limit=RETRIEVAL_LIMIT)

    if not search_results.points:
        return "Could not find any relevant chunks.", 0.0    

    # Filter results based on a similarity threshold (e.g., 0.5)
    relevant_points = [point for point in search_results.points if point.score >= SIMILARITY_THRESHOLD]

    if not relevant_points:
        return "Could not find any relevant chunks above the threshold.", 0.0

    retrieved_chunks = [point.payload["text"] for point in relevant_points]

    # The first result is the best match
    similarity_score = relevant_points[0].score

    gen_config = GenerationConfig(
        max_new_tokens=50,        
        return_full_text=False,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True
    )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""    
    Use the context to answer the question.    
    
    question: {question}

    context: {context}
    
    answer:
    """

    if groq_client:
        response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
        answer = response.choices[0].message.content
    else:
        result = qa_pipeline(prompt, generation_config=gen_config)
        answer = result[0]["generated_text"]

    return answer, similarity_score