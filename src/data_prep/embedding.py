import re
import logging
import unicodedata
import numpy as np
from typing import Any, List, Dict, Tuple

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from utils.config_loader import Loader
from backend.pinecone_utils import is_index_created

loader = Loader()
logger = logging.getLogger(__name__)
index_name = loader.get_pinecone_index_name()
batch_size = loader.get_pinecone_batch_size()

pc = Pinecone()


def normalize(vectors_list: List[float]) -> List[np.ndarray]:
    return vectors_list / np.linalg.norm(vectors_list)


def clean_vector_id(vector_id: str) -> str:
    vector_id = unicodedata.normalize('NFKD', vector_id).encode(
        'ascii', 'ignore').decode('ascii')
    vector_id = re.sub(r'[^a-zA-Z0-9?_-]', ' ', vector_id).strip().lower()

    return vector_id[:300]


def generate_embeddings(chunked_data: List[Dict[str, Any]]) -> None:
    embedding_model = OpenAIEmbeddings(
        model=loader.get_llm_embedding_model_name())

    documents = [doc["page_content"] for doc in chunked_data]
    metadata_list = [doc.get("metadata", {}) for doc in chunked_data]
    ids = [clean_vector_id(doc["metadata"].get(
        "question", f"doc_{i}")) for i, doc in enumerate(chunked_data)]

    embedded_data = embedding_model.embed_documents(documents)
    normalized_vectors = [normalize(v) for v in embedded_data]

    upload_embeddings(list(zip(ids, normalized_vectors, metadata_list)))


def upload_embeddings(embeddings: List[Tuple[str, List[np.ndarray], Dict[str, Any]]]) -> None:
    if not embeddings:
        logger.warning("No embeddings to upload.")
        return

    is_index_created()
    index = pc.Index(index_name)

    # Pinecone has a limit of 4MB per request
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]

        try:
            index.upsert(vectors=[
                {
                    "id": entry[0],
                    "values": entry[1],
                    "metadata": entry[2]
                }
                for entry in batch
            ])

            logger.info(
                f"A batch of {len(batch)} embeddings successfully uploaded to Pinecone.")
        except Exception as e:
            logger.error(f"Error uploading batch to Pinecone: {str(e)}")

    logger.info(
        f"All {len(embeddings)} embeddings successfully uploaded to Pinecone.")


def embed_data(chunked_data: List[Dict[str, Any]]) -> None:
    logger.info("Starting embedding process...")

    generate_embeddings(chunked_data)
