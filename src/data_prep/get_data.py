import os
import re
import json
import logging
from typing import Any, Dict, List

from firecrawl import FirecrawlApp
from openai import OpenAI

from utils.config_loader import Loader

loader = Loader()
logger = logging.getLogger(__name__)

chunk_size = loader.get_data_chunk_size()
fields = loader.get_data_fields()
system_prompt = loader.get_data_system_prompt()
prepare_model_name = loader.get_data_model()


def separate_text(raw_text: str) -> str:
    matches = list(re.finditer(r"\*\*", raw_text))

    if len(matches) < 2:
        return raw_text

    second_last_match = matches[-2]
    split_index = second_last_match.end()

    usable_text = raw_text[:split_index - 2].rstrip()

    return usable_text


def get_data(url: str) -> Dict[str, Any]:
    logger.info(f"Started data scrapping from {url}.")

    client = OpenAI()
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    page_content = app.scrape_url(url=url)
    page_content = page_content['markdown']

    completions = {"records": []}
    while len(page_content) > 0:
        complete_chunk_content = page_content[:chunk_size]

        # Since this page has more tokens scrapped that OpenAI can process in a single request, we need to send it on batches.
        # To do this we will select chunks of fixed lenght and look for a pattern with '** record category **' on the content.
        # This marks the beginning of a new record, avoiding to send to API a not complete record.
        if len(complete_chunk_content) == chunk_size:
            adjusted_chunk_content = separate_text(complete_chunk_content)
        else:
            adjusted_chunk_content = complete_chunk_content

        user_prompt = f"""
            The scrapped content is: {adjusted_chunk_content}
            Extract the fields: {fields}
        """

        completion = client.chat.completions.create(
            model=prepare_model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        parsed_json = json.loads(completion.choices[0].message.content)
        completions["records"].extend(parsed_json["records"])

        logger.info(
            f"Data chunk processed. {len(parsed_json['records'])} records extracted.")

        page_content = page_content[len(adjusted_chunk_content):] if len(
            page_content) > len(adjusted_chunk_content) else ""

    logger.info(
        f"Scrapping data process completed.\n{len(completions['records'])} records extracted.")

    return completions
