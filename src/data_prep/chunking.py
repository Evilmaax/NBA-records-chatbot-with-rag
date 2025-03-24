import logging
from typing import Any, Dict, List

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


def chunk_raw_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info("Starting chunking process...")

    try:
        docs = []

        for record in raw_data['records']:
            question = f"Who holds the record for {record['record_category']}?"
            date = f"Record was set on {record['date']}" if record['date'] not in [
                "N/A", "na"] else ""

            answer = (
                f"{record['holder']} holds the record for {record['record_category']} with {record['record']}. " + date).strip()

            doc = Document(
                page_content=question,
                metadata={
                    "question": question,
                    "holder": record['holder'],
                    "answer": answer,
                    "record_category": record['record_category'],
                    "record": record['record'],
                    "date": record['date'] if record['date'] != None else "N/A",
                }
            )
            docs.append(doc)

        documents_list = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]

        logger.info(f"chunking process completed!")
        return documents_list

    except Exception as e:
        logger.error(f"Error on chunking process: {str(e)}")
