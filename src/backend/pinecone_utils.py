import logging

from pinecone import Pinecone, ServerlessSpec

from utils.config_loader import Loader

loader = Loader()
logger = logging.getLogger(__name__)
index_name = loader.get_pinecone_index_name()

pc = Pinecone()


def check_vector_store() -> None:
    logger.info("Checking if vector store is created and has embeddings...")

    is_index_created()
    has_data = index_has_data()

    if not has_data:
        logger.info("This will take some time.")
        from data_prep.prepare_data import prepare_data

        prepare_data()


def is_index_created() -> None:
    if index_name not in pc.list_indexes().names():
        create_index()


def create_index() -> None:
    logger.info(
        f"'{index_name}' index unavailable. Creating a new index.")

    pc.create_index(
        name=index_name,
        dimension=loader.get_pinecone_dimension(),
        metric=loader.get_pinecone_metric(),
        spec=ServerlessSpec(
            cloud=loader.get_pinecone_cloud_provider(),
            region=loader.get_pinecone_cloud_region()
        )
    )

    logger.info(f"Pinecone index '{index_name}' created.")


def index_has_data() -> bool:
    stats = pc.Index(index_name).describe_index_stats()
    embeddings = stats.get("total_vector_count", 0)

    if embeddings > 0:
        logger.info(
            f"Vector store is ready and contains {embeddings} embeddings.")
        return True
    else:
        logger.info(
            "Vector store exists but has no embeddings. Need to populate.")
        return False
