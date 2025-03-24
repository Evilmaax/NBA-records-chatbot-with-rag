import logging
from typing import Tuple, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models import BaseLanguageModel

from backend.build_chain import build_runnable_chain
from backend.cache import find_in_cache, update_cache
from utils.config_loader import Loader

loader = Loader()
logger = logging.getLogger(__name__)
confidence_threshold = loader.get_pinecone_threshold()

high_confidence_path = RunnableLambda(
    lambda x: {
        "content": f"Confident answer: {x['context'][0]['metadata']['answer']}",
    }
)


def get_answer(user_input: str) -> Tuple[Dict[str, str], bool]:
    cache_result = find_in_cache(user_input)

    if cache_result:
        return cache_result, True

    logger.info("No cache hit. Invoking LLM...")
    llm_answer, is_cache = llm_invoke(user_input)

    return llm_answer, is_cache


def llm_invoke(user_input: str) -> Tuple[Any, bool]:
    model = ChatOpenAI(
        model_name=loader.get_llm_chat_completion_model_name(), temperature=loader.get_llm_temperature())
    chain = build_runnable_chain(model)

    try:
        llm_answer = chain.invoke({"input_message": user_input})
    except Exception as e:
        logger.error("Chain execution failed: %s", str(e))
        return {"content": "Error processing request due to chain problems"}, False

    update_cache(user_input, llm_answer)

    if isinstance(llm_answer, AIMessage):
        return llm_answer, False
    else:
        return llm_answer["content"], True


def call_llm(prompt_value: ChatPromptValue, model: BaseLanguageModel) -> AIMessage:
    try:
        return model.invoke(prompt_value.to_messages())
    except Exception as e:
        logger.error("LLM invocation failed: %s", str(e))
        return AIMessage(content="Error processing request")
