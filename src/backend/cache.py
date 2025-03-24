import os
import redis
import logging

from typing import Optional, Union
from rapidfuzz import process, fuzz
from langchain_core.messages.ai import AIMessage

from utils.config_loader import Loader


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

loader = Loader()
logger = logging.getLogger(__name__)


def find_in_cache(user_input: str) -> Optional[str]:
    cached_keys = redis_client.keys("*")
    best_match = process.extractOne(user_input, cached_keys, scorer=fuzz.ratio)

    if best_match and best_match[1] >= loader.get_redis_cache_threshold():
        score = best_match[1]
        logger.info(
            f"Match found with {score}% confidence in Redis cache! Skipping LLM invoke.")
        return redis_client.get(best_match[0])
    else:
        return None


def update_cache(user_input: str, llm_answer: Union[dict, AIMessage]) -> None:
    content = llm_answer.get("content", "") if isinstance(
        llm_answer, dict) else llm_answer.content
    redis_client.set(user_input, content, ex=86400)

    logger.info(f"Cache updated in Redis for input: {user_input}")
