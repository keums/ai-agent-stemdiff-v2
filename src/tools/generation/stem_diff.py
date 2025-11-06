import asyncio
import json
import os
import pathlib
import pprint
from typing import Any, Dict, List, Optional, Tuple

import librosa
import pyrubberband as pyrb
import soundfile as sf
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tritonclient.http import InferenceServerClient, InferRequestedOutput

from models import ContextSong
from tools.mcp_base import tool
from common.cache import cache_get
from utils.check_time import check_time
from utils.s3_audio_file import S3Audio

env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DEV_TRITON_HOST = os.getenv("DEV_TRITON_HOST")
PROD_TRITON_HOST = os.getenv("PROD_TRITON_HOST")
OUTPUT_FORMAT = "aac"
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_URI_ROOT = "s3://ai-agent-data-new/generated_stem"
OUTPUT_LOCAL_ROOT = "./output"

# 🎵 Root Note 기준 매핑 (C=1, C#/Db=2, D=3, ..., B=12)
# 메이저와 마이너는 같은 근음이면 같은 번호
KEY_MAPPING = {
    "CM": 1,
    "Cm": 1,
    "C#M": 2,
    "C#m": 2,
    "DbM": 2,
    "Dbm": 2,
    "DM": 3,
    "Dm": 3,
    "D#M": 4,
    "D#m": 4,
    "EbM": 4,
    "Ebm": 4,
    "EM": 5,
    "Em": 5,
    "FM": 6,
    "Fm": 6,
    "F#M": 7,
    "F#m": 7,
    "GbM": 7,
    "Gbm": 7,
    "GM": 8,
    "Gm": 8,
    "G#M": 9,
    "G#m": 9,
    "AbM": 9,
    "Abm": 9,
    "AM": 10,
    "Am": 10,
    "A#M": 11,
    "A#m": 11,
    "BbM": 11,
    "Bbm": 11,
    "BM": 12,
    "Bm": 12,
}

# 전역 업로더 인스턴스
_s3_uploader = None


# %% pitch shift


class GenerateStemDiffInput(BaseModel):
    """Generate stem diff tool에 대한 입력 스키마"""

    context_song_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Context song info"
    )
    prompt_stem_info: List[Dict[str, Any]] = Field(
        default=[], description="Prompt stem info"
    )
    mix_stem_diff: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of mix stem diff URIs"
    )
    task_id: str = Field(default="", description="Task ID")


class GenerateStemDiffOutput(BaseModel):
    output_uris: List[str] = Field(default=[], description="Output URIs")


def get_s3_uploader() -> S3Audio:
    """S3 업로더 싱글톤 인스턴스 반환"""
    global _s3_uploader
    if _s3_uploader is None:
        _s3_uploader = S3Audio()
    return _s3_uploader


def convert_flac_to_aac(input_path, output_path):
    import ffmpeg

    (
        ffmpeg.input(input_path)
        .output(output_path, acodec="aac", audio_bitrate="192k")
        .run(overwrite_output=True, quiet=True)
    )


# @check_time
def pitch_shift(
    original_uri: str,
    pitch_shift_semitones: int,
    upload_to_s3: bool = True,
    s3_folder: str = "test/stemdiff",
    output_format: str = "flac",
) -> dict:
    """
    오디오 파일의 피치를 변경하고 선택적으로 S3에 업로드

    Args:
        file_path (str): 입력 파일 경로
        pitch_shift_semitones (int): 피치 변경량 (세미톤)
        upload_to_s3 (bool): S3 업로드 여부
        s3_folder (str): S3 폴더 경로

    Returns:
        dict: 처리 결과 정보
    """
    file_name, file_extension = os.path.splitext(original_uri)
    base_name = file_name.split("/")[-1]
    if pitch_shift_semitones != 0:
        # !우선적으로 flac 파일 다운로드 후 피치 시프
        # download_file_path = f"{OUTPUT_LOCAL_ROOT}/{base_name}.{output_format}"
        download_file_path = f"{OUTPUT_LOCAL_ROOT}/audio/{base_name}.flac"
        print("output_local_root:", OUTPUT_LOCAL_ROOT)
        print("base_name:", base_name)
        print("original_uri:", original_uri)
        print("download_file_path:", download_file_path)
        S3Audio().download_audio_file(
            original_uri.replace("aac", "flac"), download_file_path
        )
        y, sr = librosa.load(download_file_path, sr=OUTPUT_SAMPLE_RATE)
        # print(f"y shape: {y.shape}, dtype: {y.dtype}, sr: {sr}")

        # pyrb : (samples, channels), float32
        y_shifted = pyrb.pitch_shift(
            y,
            sr,
            n_steps=pitch_shift_semitones,
            rbargs={
                "--fine": "",
                "--formant": "",
            },
        )
        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_LOCAL_ROOT, exist_ok=True)
        shifted_audio_path = f"{OUTPUT_LOCAL_ROOT}/audio/{base_name}.flac"
        sf.write(shifted_audio_path, y_shifted, sr, format="FLAC")
        # librosa.output.write_flac(shifted_audio_path, y_shifted, sr)
        if output_format == "aac":
            convert_flac_to_aac(
                shifted_audio_path, shifted_audio_path.replace(".flac", ".aac")
            )
    else:
        shifted_audio_path = ""

    if upload_to_s3:
        if pitch_shift_semitones == 0:
            s3_uri = original_uri
            print(f"🔧 Already uploaded original audio: {original_uri}")
        else:
            try:
                uploader = get_s3_uploader()
                s3_path = f"{s3_folder}/{base_name}.{output_format}"
                s3_uri = uploader.upload_audio_file(shifted_audio_path, s3_path)
                if output_format == "aac":
                    s3_path = f"{s3_folder}/{base_name}.aac"
                    s3_uri = uploader.upload_audio_file(shifted_audio_path, s3_path)

                if s3_uri:
                    print(f"✅ Pitch shifted audio uploaded to S3: {s3_uri}")
                else:
                    print("❌ S3 upload failed")

            except Exception as e:
                print(f"❌ S3 upload error: {e}")

    result = {
        "shifted_audio_local_path": shifted_audio_path,
        "original_audio_uri": original_uri,
        "shifted_audio_uri": s3_uri,
        "pitch_shift_semitones": pitch_shift_semitones,
        "success": True,
    }

    return result


def get_candidate_key_difference(
    reference_key: str, isMixMajorMinor: bool = False, keyRange: int = 6
):
    """
    주어진 키에 대한 후보 키 목록 생성 (원형 거리 계산)

    Args:
        query_key (str): 기준 키
        isMixMajorMinor (bool): 장조와 단조 혼합 여부
        keyRange (int): 키 범위 (최대 6)

    Returns:
        Tuple[List[str], Dict[str, int]]: 후보 키 목록과 차이 딕셔너리
    """
    if reference_key == "-":
        return [], {}

    reference_key_idx = KEY_MAPPING[reference_key]
    reference_key_type = reference_key[-1]  # 'M' 또는 'm'

    candidate_keys = []
    key_differences = {}

    # 🎵 keyRange는 최대 6으로 제한 (tritone이 최대 거리)
    keyRange = min(keyRange, 6)

    # 모든 가능한 키들을 순회하면서 원형 거리 계산
    for target_key, target_idx in KEY_MAPPING.items():
        # 장/단조 혼합이 허용되지 않으면 같은 타입의 키만 포함
        if not isMixMajorMinor and target_key[-1] != reference_key_type:
            continue

        # 🎵 원형 거리 계산 (12반음 체계) - 더 안전한 버전
        # raw_diff = target_idx - query_key_idx
        raw_diff = reference_key_idx - target_idx

        # 원형 구조에서 최단 거리 계산 (명확한 로직)
        if raw_diff == 0:
            circular_diff = 0
        elif raw_diff > 0:
            # 양수: 시계방향과 반시계방향 중 짧은 거리 선택
            clockwise = raw_diff
            counter_clockwise = raw_diff - 12
            circular_diff = clockwise if clockwise <= 6 else counter_clockwise
        else:
            # 음수: 반시계방향과 시계방향 중 짧은 거리 선택
            counter_clockwise = raw_diff
            clockwise = raw_diff + 12
            circular_diff = (
                counter_clockwise if abs(counter_clockwise) <= 6 else clockwise
            )

        # keyRange 내의 키만 포함
        if abs(circular_diff) <= keyRange:
            candidate_keys.append(target_key)
            key_differences[target_key] = circular_diff

    # 🔧 거리 순으로 정렬 (가까운 키가 먼저)
    candidate_keys.sort(key=lambda k: (abs(key_differences[k]), k))

    return candidate_keys, key_differences


# %% stem diff
@check_time
def stem_diff(payload, env: Optional[str] = "dev"):

    print("\n✅ **STEM_DIFF** FUNCTION CALLED!")
    output_format = payload.get("output_format", OUTPUT_FORMAT)
    payload["output_uris"] = (
        f"{OUTPUT_URI_ROOT}/{payload['task_id']}-{payload['category']}"
    )
    HOST = PROD_TRITON_HOST if env == "prod" else DEV_TRITON_HOST
    with InferenceServerClient(HOST) as client:
        print("Waiting for the response...")
        print("host:", HOST)
        response = client.infer(
            "pipeline",
            inputs=[],
            outputs=[
                InferRequestedOutput("uris"),
            ],
            parameters={
                "prompt_uris": json.dumps(payload["prompt_uris"]),
                "context_uris": json.dumps(payload["context_uris"]),
                "bpm": str(payload["bpm"]),
                "bars": int(round(payload["bars"])),
                "key": payload["key"],
                "output_format": output_format,
                "output_uri": payload["output_uris"],
                "output_sample_rate": OUTPUT_SAMPLE_RATE,
            },
            timeout=60_000_000,  # 60 seconds
        )

        uris = response.as_numpy("uris")
        uris = [uri.decode() for uri in uris]
        return uris


async def process_pitch_shift_single(
    idx: int,
    tmp_prompt_stem_info: Dict[str, Any],
    original_uri: str,
    context_song_info: Dict[str, Any],
) -> Tuple[int, str]:
    """단일 pitch shift 작업을 수행하는 헬퍼 함수 (순서 보장을 위해 인덱스 포함)"""
    if tmp_prompt_stem_info["stemType"] not in ["rhythm", "fx", "mixed"]:
        candidate_keys, key_differences_list = get_candidate_key_difference(
            reference_key=context_song_info["key"],
            isMixMajorMinor=False,
            keyRange=3,
        )
        key_difference = key_differences_list[tmp_prompt_stem_info["key"]]

        # ThreadPoolExecutor에서 pitch_shift 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # 기본 ThreadPoolExecutor 사용
            pitch_shift,
            original_uri,
            key_difference,
            True,  # upload_to_s3
            "test/stemdiff",  # s3_folder
            OUTPUT_FORMAT,  # output_format
        )
        print(f"\n📊 pitch_shift_result for stem {idx}: \n", result)
        return idx, result["shifted_audio_uri"]
    else:
        # pitch shift가 필요없는 경우 원본 URI 반환
        return idx, original_uri


async def process_stem_diff(
    prompt_stem_info: Dict[str, Any],
    context_diff_uris: List[str],
    context_song_info: Dict[str, Any],
    task_id: str,
    env: Optional[str] = "dev",
) -> str:
    """스템 diff 처리를 위한 헬퍼 함수 (병렬 pitch shift 처리)"""

    prompt_uris = [
        tmp_prompt_stem_info["uri"] + f".{OUTPUT_FORMAT}"
        for tmp_prompt_stem_info in prompt_stem_info
    ]

    payload_diff = {
        "prompt_uris": prompt_uris,
        "context_uris": context_diff_uris,
        "bpm": context_song_info["bpm"],
        "bars": context_song_info["bar_count"],
        "key": context_song_info["key"],
        "category": prompt_stem_info[0]["stemType"],
        "output_format": OUTPUT_FORMAT,
        "output_uris": [],
        "task_id": task_id,
    }

    # 병렬로 pitch shift 작업 수행
    pitch_shift_tasks = [
        process_pitch_shift_single(
            idx,
            tmp_prompt_stem_info,
            payload_diff["prompt_uris"][idx],
            context_song_info,
        )
        for idx, tmp_prompt_stem_info in enumerate(prompt_stem_info)
    ]

    # 모든 pitch shift 작업을 병렬로 실행
    pitch_shift_results = await asyncio.gather(*pitch_shift_tasks)

    # 순서대로 결과를 payload_diff["prompt_uris"]에 반영
    for idx, shifted_uri in pitch_shift_results:
        payload_diff["prompt_uris"][idx] = shifted_uri

    print("\n📊 payload_diff after parallel pitch shift")
    pprint.pprint(payload_diff)

    output_diff_uris = stem_diff(payload_diff, env=env)
    output_diff_uris = [uri.split(".")[0] for uri in output_diff_uris]
    return output_diff_uris


# deprecated
def process_single_stem_diff(
    tmp_prompt_stem_info: Dict[str, Any],
    idx: int,
    context_diff_uris: List[str],
    context_song_info: Dict[str, Any],
    task_id: str,
) -> str:
    """단일 스템 diff 처리를 위한 헬퍼 함수"""
    print("context_diff_uris:", context_diff_uris)
    payload_diff = {
        "prompt_uri": tmp_prompt_stem_info["uri"] + f".{OUTPUT_FORMAT}",
        "context_uris": context_diff_uris,
        "bpm": context_song_info["bpm"],
        "bars": context_song_info["bar_count"],
        "key": context_song_info["key"],
        "output_format": OUTPUT_FORMAT,
        "task_id": task_id,
        "song_index": idx,
    }

    if tmp_prompt_stem_info["stemType"] not in ["rhythm", "fx", "mixed"]:
        candidate_keys, key_differences_list = get_candidate_key_difference(
            reference_key=context_song_info["key"],
            isMixMajorMinor=False,
            keyRange=3,
        )
        key_difference = key_differences_list[tmp_prompt_stem_info["key"]]
        result = pitch_shift(
            original_uri=payload_diff["prompt_uri"],
            pitch_shift_semitones=key_difference,
            output_format=OUTPUT_FORMAT,
        )
        payload_diff["prompt_uri"] = result["shifted_audio_uri"]
        print(f"\n📊 pitch_shift_result for stem {idx}: \n", result)

    print(f"\n📊 payload_diff for stem {idx}")
    pprint.pprint(payload_diff)

    output_diff_uris = stem_diff(payload_diff)
    return output_diff_uris[0].split(".")[0]


def lambda_handler(event, context):
    print("event:", event)
    print("context:", context)

    dialog_uuid = event["dialog_uuid"]
    context_song_info = json.loads(cache_get(f"context_song_info:{dialog_uuid}"))
    prompt_stem_info = json.loads(cache_get(f"prompt_stem_info:{dialog_uuid}"))
    mix_stem_diff = json.loads(cache_get(f"mix_stem_diff:{dialog_uuid}"))

    generate_stem_diff_result = asyncio.run(
        generate_stem_diff(
            context_song_info=context_song_info,
            prompt_stem_info=prompt_stem_info,
            mix_stem_diff=mix_stem_diff,
            task_id=dialog_uuid,
        )
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "output_uris": generate_stem_diff_result.output_uris,
            }
        ),
    }


async def generate_stem_diff(
    context_song_info: Dict[str, Any],
    prompt_stem_info: list[Dict[str, Any]],
    mix_stem_diff: Optional[List[Dict[str, Any]]],
    task_id: str,
    env: Optional[str] = "dev",
):

    generated_mix_stem_types = []
    if mix_stem_diff:
        generated_mix_stem_types = [
            stem_info["category"] for stem_info in mix_stem_diff
        ]

        print("\n📊 Generated_mix_stem_types: \n", generated_mix_stem_types)

    #! 1. context_song은 무조건 제외
    #! 2. genrated_audio_uri 는 rhythm, low, fx, melody는 제외, mid, high는 포함

    prompt_stem_type = prompt_stem_info[0][
        "stemType"
    ]  # 일단은 같은 스템이니까 첫번째 스템 타입만 사용

    context_diff_uris = []
    for uri in context_song_info["context_audio_uris"]:
        uri_stem_type = uri.split("/")[-2]
        if (
            uri_stem_type != prompt_stem_type
            and uri_stem_type not in generated_mix_stem_types
        ):
            context_diff_uris.append(
                uri.replace("https://", "s3://").replace(".s3.amazonaws.com", "")
                + f".{OUTPUT_FORMAT}"
            )

    if mix_stem_diff:
        for stem_info in mix_stem_diff:
            context_diff_uris.append(
                stem_info["uri"]
                .replace("https://", "s3://")
                .replace(".s3.amazonaws.com", "")
                + f".{OUTPUT_FORMAT}"
            )

    print(f"\n🚀 Start stemdiff processing: {len(prompt_stem_info)} stems")
    output_uris = await process_stem_diff(
        prompt_stem_info=prompt_stem_info,
        context_diff_uris=context_diff_uris,
        context_song_info=context_song_info,
        task_id=task_id,
        env=env,
    )
    print("\n📊 output_uris: \n", output_uris)
    return GenerateStemDiffOutput(output_uris=output_uris)

    # async def process_stem_async(stem_info, idx):
    #     """단일 스템을 비동기로 처리"""
    #     loop = asyncio.get_event_loop()
    #     try:
    #         # I/O 바운드 작업을 executor에서 실행
    #         result = await loop.run_in_executor(
    #             None,  # 기본 ThreadPoolExecutor 사용
    #             process_single_stem_diff,
    #             stem_info,
    #             idx,
    #             context_diff_uris,
    #             context_song_info,
    #             task_id,
    #         )
    #         print(f"✅ Stem {idx} processing completed: {result}")
    #         return result
    #     except Exception as exc:
    #         print(f"❌ Stem {idx} processing error: {exc}")
    #         raise exc

    # # 모든 태스크를 병렬로 실행하고 순서 보장
    # tasks = [
    #     process_stem_async(stem_info, idx)
    #     for idx, stem_info in enumerate(prompt_stem_info)
    # ]

    # try:
    #     # gather는 입력 순서대로 결과를 반환
    #     output_uris = await asyncio.gather(*tasks)
    # except Exception as exc:
    #     print(f"❌ Parallel processing failed: {exc}")
    #     raise exc

    # print("\n📊 output_uris: \n", output_uris)
    # return GenerateStemDiffOutput(output_uris=output_uris)


if __name__ == "__main__":
    print("S3 URIs of generated audio:")
    # # IMPORTANT! These metadata must match the context audio.
    payload = {
        "prompt_uris": [
            "s3://ai-agent-data-new/block_data/b000001_loveback/G/high/b000001_loveback-g-high.aac",
            "s3://ai-agent-data-new/block_data/b000101_justtheintro/E/high/b000101_justtheintro-e-high.aac",
        ],
        "context_uris": [
            "s3://ai-agent-data-new/block_data/p000086_cheerupbaby/C/low/p000086_cheerupbaby-c-low.aac",
            "s3://ai-agent-data-new/block_data/p000086_cheerupbaby/C/mid/p000086_cheerupbaby-c-mid.aac",
            "s3://ai-agent-data-new/block_data/p000086_cheerupbaby/C/rhythm/p000086_cheerupbaby-c-rhythm.aac",
        ],
        "bpm": 90,
        "bars": 4,
        "key": "Gm",
        "category": "high",
        "task_id": "aaaaa",
    }

    uris = stem_diff(payload)
    print(uris)
