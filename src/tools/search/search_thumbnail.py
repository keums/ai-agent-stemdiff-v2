import asyncio
import os

import numpy as np
from pydantic import BaseModel

from tools.embedding.text_embedding import get_stem_embeddings
from tools.mcp_base import tool

from .stem_search import get_es_client


class SearchThumbnailInput(BaseModel):
    input_text: str


class SearchThumbnailOutput(BaseModel):
    thumbnail_url: str


ES_THUMBNAIL_INDEX = os.getenv("ES_THUMBNAIL_INDEX") or ""
ROOT_THUMBNAIL_URL = os.getenv("ROOT_THUMBNAIL_URL") or ""
_es_client = None


@tool(
    name="search_thumbnail_from_es",
    description="Search for a thumbnail for a given image",
    input_schema=SearchThumbnailInput,
    output_schema=SearchThumbnailOutput,
)
async def search_thumbnail(input_text: str) -> SearchThumbnailOutput:
    input_dict = [{"category": "thumbnail", "text": input_text, "uri": ""}]
    es_client = get_es_client()
    if es_client is None:
        print("Elasticsearch client not available")
        return SearchThumbnailOutput(thumbnail_url="")

    embedding = get_stem_embeddings(input_dict)[0]["embedding"]
    retriever_config = {
        "rrf": {
            "retrievers": [
                {
                    "knn": {
                        "field": "captionEmbed",
                        "query_vector": embedding,
                        "k": 10,
                        "num_candidates": 30,
                    }
                }
            ],
            "rank_window_size": 100,
            "rank_constant": 10,
        }
    }

    search_results = es_client.search(
        index=ES_THUMBNAIL_INDEX,
        retriever=retriever_config,
        _source=["id"],
        size=100,
    )
    weights = np.linspace(1.0, 0.3, len(search_results["hits"]["hits"]))
    weights = weights / np.sum(weights)  # 예: 선형 감소
    selected_thumbnail = np.random.choice(
        search_results["hits"]["hits"], size=1, replace=False, p=weights
    )[0]

    thumbnail_url = f"{ROOT_THUMBNAIL_URL}/{selected_thumbnail['_source']['id']}"
    return SearchThumbnailOutput(thumbnail_url=thumbnail_url)


if __name__ == "__main__":
    input_text = "A guitar solo with a lot of distortion and reverb"
    result = asyncio.run(search_thumbnail(input_text))
    print(result)
