import logging
import os
import uuid
from typing import Any, Dict

import requests
from pydantic import BaseModel, Field

from tools.mcp_base import tool

logger = logging.getLogger(__name__)

# API 엔드포인트 정의
CAPTIONING_API_URL = os.getenv("CAPTIONING_API_URL")

# 기본 캡션 텍스트
DEFAULT_CAPTION = "No description."


def extract_caption_from_response(result: Dict[str, Any]) -> str:
    """
    API 응답에서 캡션을 추출합니다.

    Args:
        result (Dict[str, Any]): API 응답 데이터

    Returns:
        str: 추출된 캡션 텍스트
    """
    try:
        # captioning 필드 추출
        caption_data = result["captioning"]

        # 캡션이 비어있는 경우 기본값 설정
        if not caption_data or caption_data == "":
            logger.info("Empty caption")
            return DEFAULT_CAPTION

        return caption_data

    except KeyError:
        logger.warning("KeyError: 'captioning' field not found in API response")
        # 다른 형식의 API 응답 처리 시도
        if "captionEmbed" in result:
            logger.info("No caption text, but found embedding")
            return DEFAULT_CAPTION

        if "caption" in result:
            return result["caption"]

        return DEFAULT_CAPTION


# 캡셔닝 함수
def get_music_caption(audio_file_path: str, target_language: str = "en") -> str:
    """
    오디오 파일로부터 음악 설명을 생성합니다.

    Args:
        audio_file_path (str): 오디오 파일 경로
        target_language (str): 캡션 생성 대상 언어 (기본값: 'en')

    Returns:
        str: 생성된 캡션 텍스트
    """
    try:
        # 파일 경로 확인
        if not os.path.exists(audio_file_path):
            logger.error(f"File not found: {audio_file_path}")
            return "Error: File not found"

        # 파일 읽기
        with open(audio_file_path, "rb") as audio_file:
            file_content = audio_file.read()
            logger.info(
                f"Read file: {audio_file_path}, size: {len(file_content)} bytes"
            )

        # 고유한 파일명 생성
        original_filename = os.path.basename(audio_file_path)
        unique_filename = f"{uuid.uuid4().hex}.mp3"

        logger.info(f"Using filename: {unique_filename}")

        # API 호출
        files = {"file": (unique_filename, file_content)}
        data = {"targetLanguage": target_language}

        logger.info(f"Calling captioning API for {original_filename}")
        response = requests.post(CAPTIONING_API_URL, files=files, data=data)

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            return f"Error: API returned status {response.status_code}"

        # 결과 파싱
        result = response.json()
        logger.info(f"API response keys: {list(result.keys())}")

        # 캡션 추출
        caption = extract_caption_from_response(result)
        logger.info(f"Caption: {caption[:100]}")

        return caption

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Error: {str(e)}"


# 비동기 캡셔닝 함수
async def get_music_caption_async(
    audio_file_path: str, target_language: str = "en"
) -> str:
    """
    오디오 파일로부터 비동기적으로 음악 설명을 생성합니다.

    Args:
        audio_file_path (str): 오디오 파일 경로
        target_language (str): 캡션 생성 대상 언어 (기본값: 'en')

    Returns:
        str: 생성된 캡션 텍스트
    """
    try:
        import aiohttp

        # 파일 경로 확인
        if not os.path.exists(audio_file_path):
            logger.error(f"File not found: {audio_file_path}")
            return "Error: File not found"

        # 파일 읽기
        with open(audio_file_path, "rb") as audio_file:
            file_content = audio_file.read()
            logger.info(
                f"Read file: {audio_file_path}, size: {len(file_content)} bytes"
            )

        # 고유한 파일명 생성
        original_filename = os.path.basename(audio_file_path)
        unique_filename = f"{uuid.uuid4().hex}.mp3"
        logger.info(f"Using filename: {unique_filename}")

        # API 요청 준비
        data = aiohttp.FormData()
        data.add_field("file", file_content, filename=unique_filename)
        data.add_field("targetLanguage", target_language)

        logger.info(f"Async calling captioning API for {original_filename}")

        # 비동기 API 호출
        async with aiohttp.ClientSession() as session:
            async with session.post(CAPTIONING_API_URL, data=data) as response:
                if response.status != 200:
                    logger.error(f"API error: {response.status}")
                    return f"Error: API returned status {response.status}"

                try:
                    result = await response.json()
                    logger.info(f"API response keys: {list(result.keys())}")
                    # logger.info(
                    #     f"API full response: {json.dumps(result, ensure_ascii=False)}"
                    # )
                    # 캡션 추출
                    caption = extract_caption_from_response(result)
                    logger.info(f"Caption: {caption}")

                    return caption

                except Exception as e:
                    logger.error(f"Failed to process response: {e}")
                    return f"Error: {str(e)}"

    except Exception as e:
        logger.error(f"Error in get_music_caption_async: {e}")
        return f"Error: {str(e)}"


# 입력 스키마 정의
class MusicCaptioningInput(BaseModel):
    """
    음악 캡셔닝 도구에 대한 입력 스키마

    오디오 파일의 경로와 대상 언어를 지정하여 캡션 생성에 필요한 파라미터를 제공합니다.
    """

    audio_path: str = Field(
        ..., description="Path to the audio file to generate caption for"
    )
    target_language: str = Field(
        default="en", description="Target language for caption (default: English)"
    )


# 출력 스키마 정의
class MusicCaptioningOutput(BaseModel):
    """
    음악 캡셔닝 도구에 대한 출력 스키마

    생성된 음악 캡션 텍스트를 포함합니다.
    """

    caption: str = Field(
        description="Generated caption text describing the music content"
    )
    status: str = Field(
        default="success", description="Processing status (success/error)"
    )


@tool(
    name="get_caption_from_music",
    description="Generates descriptive text caption for music audio files",
    input_schema=MusicCaptioningInput,
    output_schema=MusicCaptioningOutput,
)
async def get_caption_from_music(params: Dict[str, Any]) -> MusicCaptioningOutput:
    """
    오디오 파일에서 음악 설명(캡션)을 생성합니다.

    Args:
        params: Dictionary containing all parameters:
            - audio_path: 오디오 파일 경로
            - target_language: 생성할 캡션의 언어 (기본값: "en")
            - task_id: 작업 식별자

    Returns:
        음악 설명(캡션) 텍스트
    """
    print("✅ **GET_CAPTION_FROM_MUSIC** FUNCTION CALLED!")
    # Pydantic 스키마를 통한 안전한 파라미터 처리
    try:
        # 먼저 스키마로 유효성 검사 및 파라미터 정규화
        input_data = MusicCaptioningInput(**params)
        logger.info(f"MusicCaptioning tool processing for task {input_data.task_id}")
    except Exception as e:
        logger.warning(f"Input validation failed, using default values: {str(e)}")
        # 유효성 검사 실패 시 안전한 기본값 사용
        input_data = MusicCaptioningInput(
            audio_path=params.get("audio_path", ""),
            target_language=params.get("target_language", "en"),
            task_id=params.get("task_id"),
        )

    # 검증된 데이터에서 값 추출
    audio_path = input_data.audio_path
    target_language = input_data.target_language or "en"
    task_id = input_data.task_id

    logger.info(f"Processing task {task_id}: {audio_path}")

    try:
        # 비동기 방식으로 캡션 생성
        caption_text = await get_music_caption_async(audio_path, target_language)

        # 오류 확인
        if caption_text.startswith("Error:"):
            logger.error(f"Error: {caption_text}")
            return MusicCaptioningOutput(
                caption="Error generating caption",
                status="error",
            )

        logger.info(f"Success: {caption_text[:50]}...")

        return MusicCaptioningOutput(
            caption=caption_text,
            status="success",
        )

    except Exception as e:
        logger.error(f"Exception: {str(e)}")

        return MusicCaptioningOutput(
            caption=f"Error processing audio: {str(e)}",
            status="error",
        )


# S3에서 파일 다운로드 (Lambda 환경)
def download_from_s3(s3_path: str) -> str:
    """
    S3에서 파일을 다운로드합니다.

    Args:
        s3_path (str): s3://bucket/key 형식의 S3 경로

    Returns:
        str: 다운로드된 로컬 파일 경로
    """
    try:
        from urllib.parse import urlparse

        import boto3

        s3_url = urlparse(s3_path)
        bucket = s3_url.netloc
        key = s3_url.path.lstrip("/")

        # 임시 파일 경로 생성
        temp_path = f"/tmp/{os.path.basename(key)}"

        # S3에서 다운로드
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, temp_path)

        logger.info(f"Downloaded from S3: {s3_path} -> {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"S3 download error: {e}")
        raise
