import logging

from data_prep.get_data import get_data
from data_prep.chunking import chunk_raw_data
from data_prep.embedding import embed_data
from backend.pinecone_utils import is_index_created

from utils.config_loader import Loader

loader = Loader()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)
url = loader.get_data_scrapping_url()


def prepare_data() -> None:
    logger.info("Preparing data...")

    raw_data = get_data(url)
    chunked_data = chunk_raw_data(raw_data)
    embed_data(chunked_data)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
