import os
import yaml
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "configs" / "config.yml"


class Loader:
    def __init__(self):
        try:
            with config_path.open("r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file) or {}

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            os._exit(1)

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {str(e)}")
            os._exit(1)

    def get_llm_embedding_model_name(self) -> str:
        return self.config['llm_options']['embedding_model_name']

    def get_llm_chat_completion_model_name(self) -> str:
        return self.config['llm_options']['chat_completion_model_name']

    def get_llm_temperature(self) -> int:
        return self.config['llm_options']['temperature']

    def get_llm_prompt(self) -> str:
        return self.config['llm_options']['prompt']['system_message']

    def get_pinecone_index_name(self) -> str:
        return self.config['pinecone_options']['index_name']

    def get_pinecone_dimension(self) -> int:
        return self.config['pinecone_options']['dimension']

    def get_pinecone_metric(self) -> str:
        return self.config['pinecone_options']['metric']

    def get_pinecone_cloud_provider(self) -> str:
        return self.config['pinecone_options']['cloud_provider']

    def get_pinecone_cloud_region(self) -> str:
        return self.config['pinecone_options']['cloud_region']

    def get_pinecone_threshold(self) -> float:
        return self.config['pinecone_options']['threshold']

    def get_pinecone_top_k(self) -> int:
        return self.config['pinecone_options']['top_k_results']

    def get_pinecone_batch_size(self) -> int:
        return self.config['pinecone_options']['batch_size']

    def get_data_chunk_size(self) -> int:
        return self.config['get_data_options']['chunk_size']

    def get_data_scrapping_url(self) -> str:
        return self.config['get_data_options']['scrapping_url']

    def get_data_fields(self) -> List:
        return self.config['get_data_options']['fields']

    def get_data_system_prompt(self) -> str:
        return self.config['get_data_options']['system_prompt']

    def get_data_model(self) -> int:
        return self.config['get_data_options']['model']

    def get_redis_cache_threshold(self) -> int:
        return self.config['redis_options']['redis_cache_threshold']

    def get_firebase_project_id(self) -> str:
        return self.config['firebase_options']['project_id']

    def get_firebase_chat_collection_name(self) -> str:
        return self.config['firebase_options']['chat_collection_name']

    def get_firebase_session_id_chat_history(self) -> str:
        return self.config['firebase_options']['session_id_chat_history']

    def get_firebase_llm_usage_collection_name(self) -> str:
        return self.config['firebase_options']['llm_usage_collection_name']
