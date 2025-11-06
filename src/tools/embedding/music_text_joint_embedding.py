import os
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tools.mcp_base import tool

# .env에서 읽어오기
MUSIC_TEXT_EMBEDDING_API_URL = os.getenv("MUSIC_TEXT_EMBEDDING_API_URL")
TEXT_MUSIC_EMBEDDING_API_URL = os.getenv("TEXT_MUSIC_EMBEDDING_API_URL")


class MusicTextEmbeddingInput(BaseModel):
    """음악-텍스트 임베딩 도구에 대한 입력 스키마"""

    audio_path: Optional[str] = Field(
        default=None, description="Path to the audio file to generate embeddings for"
    )
    text_prompts: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of stem prompts for batch embedding",
    )


# 출력 스키마 정의
class MusicTextEmbeddingOutput(BaseModel):
    """음악-텍스트 임베딩 도구에 대한 출력 스키마"""

    music_text_joint_embedding: List[Dict[str, Any]] = Field(
        description="Cross-modal embedding vector or dictionary of vectors"
    )


# 비동기 음악-텍스트 임베딩 함수
async def get_embedding_async(
    audio_path: Optional[str] = None,
    input_text: Optional[str] = None,
) -> List[float]:
    """
    오디오 파일과 텍스트로부터 비동기적으로 크로스모달 임베딩 벡터를 생성합니다.
    """
    try:
        import aiohttp

        # 입력 값에 따른 API uri 결정
        if input_text is None and audio_path is not None:
            # 오디오만 있는 경우 - 음악->텍스트 임베딩
            api_url = MUSIC_TEXT_EMBEDDING_API_URL

            # 오디오 파일 처리 및 FormData 준비
            if not os.path.exists(audio_path):
                print(f"Error: File not found: {audio_path}")
                return []

            with open(audio_path, "rb") as audio_file:
                file_content = audio_file.read()

            data = aiohttp.FormData()
            unique_filename = f"{uuid.uuid4().hex}.mp3"
            data.add_field("file", file_content, filename=unique_filename)

            # 비동기 API 호출 (FormData)
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, data=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    embedding_vector = result.get("mtrppembed", [])

                    return embedding_vector

        elif input_text is not None and audio_path is None:
            # 텍스트만 있는 경우 - JSON 형식으로 전송
            api_url = TEXT_MUSIC_EMBEDDING_API_URL
            # JSON 요청 준비
            headers = {"Content-Type": "application/json"}
            json_data = {"prompt": input_text}

            # 비동기 API 호출 (JSON)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url, json=json_data, headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    embedding_vector = result.get("mtrppembed", [])

                    return embedding_vector

        else:
            # 두 입력이 모두 있는 경우 (FormData + prompt 파라미터)
            if input_text is not None and audio_path is not None:
                print(
                    "Both audio and text provided, using music-to-text API as default"
                )
                api_url = MUSIC_TEXT_EMBEDDING_API_URL

                # 파일 확인 및 읽기
                if not os.path.exists(audio_path):
                    print(f"Error: File not found: {audio_path}")
                    return []

                with open(audio_path, "rb") as audio_file:
                    file_content = audio_file.read()

                # FormData 준비
                data = aiohttp.FormData()
                unique_filename = f"{uuid.uuid4().hex}.mp3"
                data.add_field("file", file_content, filename=unique_filename)
                data.add_field("prompt", input_text)  # "prompt" 파라미터 사용

                # 비동기 API 호출
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, data=data) as response:
                        response.raise_for_status()
                        result = await response.json()
                        embedding_vector = result.get("mtrppembed", [])

                        return embedding_vector
            else:
                print("Error: Neither audio nor text provided")
                return []

    except Exception as e:
        error_msg = f"Error in get_embedding_async: {str(e)}"
        print(error_msg)
        print(f"API URL: {api_url if 'api_url' in locals() else 'unknown'}")
        return []


# 스템 프롬프트에 대한 임베딩 생성 함수 추가
async def get_stem_embeddings(
    text_prompts: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    스템 프롬프트에 대한 음악-텍스트 임베딩 벡터 생성

    Args:
        text_prompts (List[Dict[str, str]]): 스템 프롬프트 리스트

    Returns:
        List[Dict[str, Any]]: 임베딩이 포함된 스템 프롬프트 리스트
    """
    embedding_outputs = []

    # 각 스템 프롬프트에 대해 개별적으로 임베딩 생성 (배치 처리 미지원)
    for embedding_input in text_prompts:
        try:
            if embedding_input["uri"] == "":
                audio_path = None

            # 텍스트 임베딩 생성 (기존 함수 재사용)
            embedding_vector = await get_embedding_async(
                audio_path=audio_path, input_text=embedding_input["text"]
            )

            # 결과 저장
            embedding_outputs.append(
                {
                    "category": embedding_input["category"],
                    "text": embedding_input["text"],
                    "uri": embedding_input["uri"],
                    "embedding": embedding_vector,
                }
            )

        except Exception as e:
            print(f"Error processing stem {embedding_input['category']}: {str(e)}")
            # 오류 시 기본값 사용
            embedding_outputs.append(
                {
                    "category": embedding_input["category"],
                    "text": embedding_input["text"],
                    "uri": embedding_input["uri"],
                    "embedding": None,
                }
            )
    return embedding_outputs


@tool(
    name="get_music_text_joint_embedding",
    description="Generates cross-modal embedding vectors between music and text",
    input_schema=MusicTextEmbeddingInput,
    output_schema=MusicTextEmbeddingOutput,
)
async def get_music_text_joint_embedding(params) -> MusicTextEmbeddingOutput:
    """
    Generate cross-modal embeddings between music and text

    Args:
        params: Dictionary containing all parameters:
            - text_prompts: List of stem prompts for batch embedding
            - task_id: Task identifier

    Returns:
        Cross-modal embedding results
    """
    print("\n✅ **GET_MUSIC_TEXT_JOINT_EMBEDDING** FUNCTION CALLED!")
    # print("🎵 get_music_text_joint_embedding called with params:", params)

    # 검증된 데이터에서 값 추출
    text_prompts = params.get("text_prompts", [])
    # audio_path = params.get("audio_path", None)

    try:
        # 입력 타입에 따라 처리 분기
        if text_prompts and isinstance(text_prompts, list):
            music_text_joint_embeddings = await get_stem_embeddings(text_prompts)
            return MusicTextEmbeddingOutput(
                music_text_joint_embedding=music_text_joint_embeddings
            )

    except Exception as e:
        error_msg = f"Error in music_text_joint_embedding: {str(e)}"
        print(error_msg)


if __name__ == "__main__":
    import asyncio

    async def test_embedding():
        result = await get_music_text_joint_embedding(
            {"text_prompts": [{"category": "test", "text": "test", "uri": ""}]}
        )
        print(result)

    asyncio.run(test_embedding())
