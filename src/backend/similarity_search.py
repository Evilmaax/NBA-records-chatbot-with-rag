import logging
from typing import Dict, List, Any

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from utils.config_loader import Loader
from data_prep.embedding import normalize

loader = Loader()
logger = logging.getLogger(__name__)
index_name = loader.get_pinecone_index_name()

pc = Pinecone()


def search_and_respond(user_query: str) -> List[Dict[str, Any]]:
    index = pc.Index(index_name)

    query_text = user_query
    embeddings = OpenAIEmbeddings(model=loader.get_llm_embedding_model_name())

    query_embedding = embeddings.embed_query(query_text)
    normalized_query = normalize(query_embedding)

    response = index.query(
        vector=normalized_query.tolist(),
        top_k=loader.get_pinecone_top_k(),
        include_metadata=True
    )

    if response.matches and response.matches[0].score > loader.get_pinecone_threshold():
        logger.info(
            f"Returning response from the vector store. Confidence: {response.matches[0].score}")

        return [{
                "metadata": response.matches[0].metadata,
                "score": response.matches[0].score
                }]
    else:
        return [
            {"metadata": {
                "Record": i + 1,
                "Category": doc['metadata'].get('record_category', 'N/A'),
                "Holder(s)": doc['metadata'].get('holder', 'N/A'),
                "Record Value": doc['metadata'].get('record', 'N/A'),
                "Date": doc['metadata'].get('date', 'N/A')},
             "score": doc.score
             }
            for i, doc in enumerate(response.matches)
        ]
