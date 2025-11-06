import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

# MCP 스타일 임포트 (환경변수 자동 로드됨)
from tools.mcp_base import tool

# from utils.check_time import check_time

logger = logging.getLogger(__name__)

# OpenAI 클라이언트 임포트 (Lambda에서는 layer로 제공)
try:
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI can't import the client.")
    OpenAI = None

# # 기본 임베딩 모델 및 차원 정의
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSION = 1536


# 입력 스키마 정의
class TextEmbeddingInput(BaseModel):
    """Text embedding tool에 대한 입력 스키마"""

    text_prompts: List[Dict[str, str]] = Field(
        description="Stem prompts for embedding (dictionary format)"
    )


class TextEmbeddingOutput(BaseModel):
    """Text embedding tool에 대한 출력 스키마"""

    text_embedding: List[Dict[str, Any]] = Field(
        description="Embedding vectors for inputs (dictionary or list format)"
    )


# # 스템 프롬프트에 대한 임베딩 생성 함수
def get_stem_embeddings(
    text_prompts: List[Dict[str, str]],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> List[Dict[str, Any]]:
    """
    스템 프롬프트에 대한 임베딩 벡터 생성

    Args:
        text_prompts (List[Dict[str, str]]): 스템 프롬프트 리스트
        model (str): 사용할 임베딩 모델
        dimension (int): 임베딩 벡터 차원

    Returns:
        List[Dict[str, Any]]: 임베딩이 포함된 스템 프롬프트 리스트
    """
    # OpenAI 클라이언트가 없는 경우 (개발/테스트 환경)
    if OpenAI is None:
        logger.warning("OpenAI 클라이언트가 없어 더미 임베딩을 반환합니다.")
        # text_prompts format : [{stemCategory:"", text:"",uri:"", embedding:[...]},
        #                       {...}
        #                       ]
        # uri is not used for embedding (for dataSchema)
        return [
            {
                "category": embedding_input["category"],
                "uri": embedding_input["uri"],
                "text": embedding_input["text"],
                "embedding": [0.1] * dimension,
            }
            for embedding_input in text_prompts
        ]

    try:
        # 배치 처리를 위한 준비
        stem_types = []
        texts = []

        # 스템 유형과 프롬프트 텍스트 추출
        for embedding_input in text_prompts:
            stem_types.append(embedding_input["category"])
            texts.append(embedding_input["text"])
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # 배치 임베딩 요청
        response = client.embeddings.create(
            model=model, input=texts, dimensions=dimension
        )

        # 결과를 딕셔너리로 변환
        result_dict = []
        for data in response.data:
            index = data.index
            if index < len(stem_types):
                result_dict.append(
                    {
                        "category": stem_types[index],
                        "uri": text_prompts[index]["uri"],
                        "text": text_prompts[index]["text"],
                        "embedding": data.embedding,
                    }
                )
        return result_dict

    except Exception as e:
        error_msg = f"OpenAI batch embedding error: {str(e)}"
        logger.error(error_msg)
        # 오류 발생 시 더미 임베딩 반환
        return [
            {
                "category": item["category"],
                "uri": item.get("uri", ""),
                "text": item["text"],
                "embedding": [0.1] * dimension,
            }
            for item in text_prompts
        ]


# @check_time
@tool(
    name="get_text_embedding",
    description="Generates text embeddings for the given input text",
    input_schema=TextEmbeddingInput,
    output_schema=TextEmbeddingOutput,
)
def get_text_embedding(params: Dict[str, Any]) -> TextEmbeddingOutput:
    """
    Process text embedding for either text_prompts (dictionary)

    Args:
        params: Dictionary containing all parameters:
            - text_prompts: List of stem prompts to embed

    Returns:
        Dictionary with embedding results
    """
    print("\n✅ **GET_TEXT_EMBEDDING** FUNCTION CALLED!")

    # 검증된 파라미터에서 값 추출
    text_prompts = params.get("text_prompts", [])

    # 두 가지 입력 형태 확인 - List[Dict[str, str]] 형태로 수정
    if text_prompts and isinstance(text_prompts, list):
        # 리스트 형태의 입력 처리
        try:
            embeddings = get_stem_embeddings(text_prompts)
            return TextEmbeddingOutput(text_embedding=embeddings)
        except Exception as e:
            logger.error(f"Error in text_prompts embedding: {str(e)}")
            return TextEmbeddingOutput(text_embedding=[])


if __name__ == "__main__":
    print(
        get_text_embedding(
            {
                "text_prompts": [
                    {"category": "test", "text": "test", "uri": ""},
                    {"category": "test2", "text": "test2", "uri": ""},
                ]
            }
        )
    )
