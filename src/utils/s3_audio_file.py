# %%
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class S3Audio:
    """S3 오디오 파일 업로드 클래스"""

    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "mixaudio-assets")
        self.download_bucket_name = os.getenv(
            "S3_DOWNLOAD_BUCKET_NAME", "ai-agent-data-new"
        )
        self.logger = logging.getLogger(__name__)

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        S3 URI를 파싱하여 버킷명과 객체 키를 반환

        Args:
            s3_uri (str): S3 URI (s3://bucket-name/object-key 형식)

        Returns:
            tuple[str, str]: (bucket_name, object_key)
        """
        if s3_uri.startswith("s3://"):
            # s3:// 제거
            path = s3_uri[5:]
            # 첫 번째 '/'를 찾아서 버킷명과 키 분리
            if "/" in path:
                bucket_name, object_key = path.split("/", 1)
                return bucket_name, object_key
            else:
                return path, ""
        else:
            # S3 URI가 아닌 경우 기본 버킷 사용
            return self.download_bucket_name, s3_uri

    def upload_audio_file(self, local_path: str, s3_path: str) -> Optional[str]:
        """
        오디오 파일을 S3에 업로드

        Args:
            local_path (str): 로컬 파일 경로
            s3_path (str): S3 저장 경로

        Returns:
            Optional[str]: 업로드된 파일의 S3 URL, 실패시 None
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            s3_url = f"s3://{self.bucket_name}/{s3_path}"
            self.logger.info(f"Upload complete: {s3_url}")
            return s3_url
        except ClientError as e:
            self.logger.error(f"Upload error: {local_path} - {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected upload error: {local_path} - {e}")
            return None

    def download_audio_file(self, s3_path: str, local_path: str) -> None:
        """
        S3에서 오디오 파일 다운로드

        Args:
            s3_path (str): S3 경로 또는 S3 URI (s3://bucket-name/object-key)
            local_path (str): 로컬 저장 경로
        """
        try:
            # S3 URI 파싱
            bucket_name, object_key = self._parse_s3_uri(s3_path)

            self.logger.info(
                f"Downloading from bucket: {bucket_name}, key: {object_key}"
            )
            self.s3_client.download_file(bucket_name, object_key, local_path)
            self.logger.info(f"Download complete: {local_path}")
        except Exception as e:
            self.logger.error(f"Download error: {s3_path} - {e}")
            raise


if __name__ == "__main__":
    s3_audio = S3Audio()
    s3_audio.download_audio_file(
        "s3://ai-agent-data-new/block_data/b001311_anything/L/high/b001311_anything-l-high.aac",
        "/Users/keums/project/ai-agent-stemdiff/output/b001311_anything-l-high.aac",
    )

# %%
