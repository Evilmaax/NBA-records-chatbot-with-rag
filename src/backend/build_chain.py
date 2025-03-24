import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableMap, RunnableBranch, Runnable

from utils.config_loader import Loader
from backend.similarity_search import search_and_respond

loader = Loader()
logger = logging.getLogger(__name__)
confidence_threshold = loader.get_pinecone_threshold()

high_confidence_path = RunnableLambda(
    lambda x: {
        "content": f"{x['context'][0]['metadata']['answer']}",
    }
)


def build_runnable_chain(model: ChatOpenAI) -> Runnable:
    from backend.llm_invoke import call_llm

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", loader.get_llm_prompt()),
        ("human", "{input_message}")
    ])

    get_vectors = RunnableMap({
        "context": RunnableLambda(lambda x: search_and_respond(x["input_message"])),
        "input_message": RunnableLambda(lambda x: x["input_message"])
    })

    check_path = RunnableBranch(
        (lambda x: x["context"][0]["score"] >=
         confidence_threshold, high_confidence_path),
        (lambda x: x["context"][0]["score"] < confidence_threshold,
         prompt_template | RunnableLambda(lambda x: call_llm(x, model))),
        RunnableLambda(
            lambda x: {"result": "No valid path chosen", "error": True})
    )

    return get_vectors | check_path
