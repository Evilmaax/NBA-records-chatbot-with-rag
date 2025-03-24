import logging
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.firebase_connection import FirebaseConnection
from backend.llm_invoke import get_answer
from backend.pinecone_utils import check_vector_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    check_vector_store()
except Exception as e:
    logger.error(f"Error with vector store: {str(e)}")
    exit(1)

firebase = None
try:
    logger.info("initializing Firebase client...")
    firebase = FirebaseConnection()
    logger.info("API is running!")
except Exception as e:
    logger.error(f"Error initializing Firebase client: {str(e)}")
    exit(1)


class InputRequest(BaseModel):
    input: str


@app.post("/ask")
def ask_question(request: InputRequest) -> Dict[str, str]:
    user_input = request.input

    try:
        if firebase:
            firebase.chat_history.add_user_message(user_input)

        llm_answer, is_cache = get_answer(user_input)

        if firebase and not is_cache:
            firebase.chat_history.add_ai_message(llm_answer)
            firebase.insert_llm_usage(llm_answer)

        else:
            logger.info(
                "Skipping LLM usage logging due to cache or high confidence")

        return {"answer": llm_answer.content if hasattr(llm_answer, "content") else llm_answer}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")
