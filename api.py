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

from fastapi import APIRouter, UploadFile, File, HTTPException
from schema import QuestionRequest, UploadResponse, AskResponse
import os

# Import your RAG logic
from rag_engine import process_document, ask_question

router = APIRouter()

UPLOAD_DIR = "./data/documents"

# Ensure upload folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Upload TXT file
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process document (chunking + embeddings + store in vector DB)
        process_document(file_path)

        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ask question
@router.post("/ask")
def ask(request: QuestionRequest):

    try:
        answer, similarity = ask_question(request.question)

        # Safety threshold
        threshold = 0.50

        if similarity < threshold:
            return AskResponse(
                answer="I could not find relevant information in the uploaded document.",
                similarity_score=similarity
            )

        return AskResponse(
            answer=answer,
            similarity_score=similarity
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))