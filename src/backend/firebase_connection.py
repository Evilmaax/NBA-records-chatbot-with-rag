import os
import logging
from datetime import datetime

import firebase_admin
from google.cloud import firestore
from firebase_admin import credentials
from langchain_core.messages.ai import AIMessage
from google.oauth2.service_account import Credentials
from langchain_google_firestore import FirestoreChatMessageHistory

from utils.config_loader import Loader

loader = Loader()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(
    BASE_DIR, "..", "configs", "nba-records-firebase-keys.json")

cred = credentials.Certificate(CREDENTIALS_PATH)
firestore_credentials = Credentials.from_service_account_file(CREDENTIALS_PATH)


class FirebaseConnection:
    def __init__(self):
        firebase_admin.initialize_app(cred)
        logger.info("Firebase started!")

        self.client = firestore.Client(
            credentials=firestore_credentials, project=loader.get_firebase_project_id())

        logger.info("Initializing Firestore Chat Message History...")
        self.chat_history = FirestoreChatMessageHistory(
            collection=loader.get_firebase_chat_collection_name(),
            session_id=loader.get_firebase_session_id_chat_history(),
            client=self.client,
        )

        logger.info("Chat History Initialized.")
        logger.debug(f"Current Chat History: {self.chat_history.messages}")

    def insert_llm_usage(self, answer: AIMessage) -> None:
        timestamp = datetime.now()
        doc_id = timestamp.strftime("%Y%m%d%H%M%S%f")

        collection_name = loader.get_firebase_llm_usage_collection_name()

        usage_ref = self.client.collection(
            collection_name).document(doc_id)

        usage_data = {
            "timestamp": timestamp,
            "model_name": answer.response_metadata.get("model_name"),
            "finish_reason": answer.response_metadata.get("finish_reason"),
            "system_fingerprint": answer.response_metadata.get("system_fingerprint"),
            "total_tokens": answer.response_metadata.get("token_usage", {}).get("total_tokens"),
            "completion_tokens": answer.response_metadata.get("token_usage", {}).get("completion_tokens"),
            "prompt_tokens": answer.response_metadata.get("token_usage", {}).get("prompt_tokens"),
            "completion_accepted_prediction_tokens": answer.response_metadata.get("token_usage", {})
            .get("completion_tokens_details", {})
            .get("accepted_prediction_tokens"),
            "completion_audio_tokens": answer.response_metadata.get("token_usage", {})
            .get("completion_tokens_details", {})
            .get("audio_tokens"),
            "completion_reasoning_tokens": answer.response_metadata.get("token_usage", {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens"),
            "completion_rejected_prediction_tokens": answer.response_metadata.get("token_usage", {})
            .get("completion_tokens_details", {})
            .get("rejected_prediction_tokens"),
            "prompt_audio_tokens": answer.response_metadata.get("token_usage", {})
            .get("prompt_tokens_details", {})
            .get("audio_tokens"),
            "prompt_cached_tokens": answer.response_metadata.get("token_usage", {})
            .get("prompt_tokens_details", {})
            .get("cached_tokens"),
        }

        usage_ref.set(usage_data)
        logger.debug(f"LLM usage logged on Firebase: {usage_data}")
