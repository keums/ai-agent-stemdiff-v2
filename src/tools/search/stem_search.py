import os
import traceback
from socket import create_server
from typing import Any, Dict, List, Optional

import numpy as np
from elasticsearch import Elasticsearch  # 동기 클라이언트
from pydantic import BaseModel, Field

from models import ContextSong, Stem
from tools.mcp_base import tool

ES_BLOCK_INDEX = os.getenv("ES_BLOCK_INDEX") or ""
ES_SONG_INDEX = os.getenv("ES_SONG_INDEX") or ""
ES_USER_BLOCK_INDEX = os.getenv("ES_USER_BLOCK_INDEX") or ""
ES_USER_SONG_INDEX = os.getenv("ES_USER_SONG_INDEX") or ""
ES_ENVIRONMENTAL_SOUND_INDEX = os.getenv("ES_ENVIRONMENTAL_SOUND_INDEX") or ""
REMIX_ONLY_TENANT = ["artist"]
# 전역 ES 클라이언트 (람다 컨테이너 재사용을 위한 싱글톤)
_es_client = None

SOURCE_LIST = [
    "id",
    "key",
    "bpm",
    "stemType",
    "barCount",
    "sectionName",
    "sectionRole",
    "songId",
    "caption",
]

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


# %%
# SearchStem 입출력 스키마
class SearchStemInput(BaseModel):
    """스템 검색 도구에 대한 입력 스키마"""

    text_embedding: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Text embedding list with embeddings"
    )
    music_text_joint_embedding: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Music text embedding list with embeddings"
    )
    music_embedding: Optional[List[Dict[str, Any]]] = (
        Field(default=None, description="Music embedding list with embeddings"),
    )
    context_song_info: Optional[Dict[str, Any]] = (
        Field(default={}, description="Context song info"),
    )
    target_music_info: Optional[Dict[str, Any]] = (
        Field(default={}, description="Target music info"),
    )
    mix_stem_diff: Optional[List[List[Dict[str, Any]]]] = Field(
        default=None, description="Mix stem diff"
    )
    request_type: Optional[str] = Field(default=None, description="Request type")
    continue_stem_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Continue song info"
    )
    request_type: Optional[str] = Field(default=None, description="Request type")
    continue_stem_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Continue song info"
    )
    remix_song_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Remix song info"
    )


class SearchStemOutput:
    """스템 검색 도구에 대한 출력 스키마"""

    prompt_stem_info: List[Dict[str, Any]] = Field(
        default=None, description="Prompt stem info"
    )
    context_song_info: Optional[ContextSong] = Field(
        default=None, description="Context song info"
    )
    vocal_stem_info: List[Dict[str, Any]] = Field(
        default=None, description="Vocal stem info"
    )

    def __init__(self, prompt_stem_info, context_song_info, vocal_stem_info):
        self.prompt_stem_info = prompt_stem_info
        self.context_song_info = context_song_info
        self.vocal_stem_info = vocal_stem_info

    def to_dict(self):
        return {
            "prompt_stem_info": self.prompt_stem_info,
            "context_song_info": self.context_song_info,
            "vocal_stem_info": self.vocal_stem_info,
        }


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


def cleanup_es_client():
    """
    ES 클라이언트 정리 (람다 종료 시 호출 권장)
    메모리 사용량 최적화
    """
    global _es_client
    if _es_client is not None:
        try:
            _es_client.close()
            print("🧹 Elasticsearch client closed")
        except Exception as e:
            print(f"⚠️ Error closing ES client: {e}")
        finally:
            _es_client = None


def warm_up_es_client():
    """
    ES 클라이언트 웜업 (람다 시작 시 호출 권장)
    첫 번째 요청 지연시간 감소
    """
    client = get_es_client()
    if client:
        try:
            # 간단한 인덱스 존재 확인으로 웜업
            client.indices.exists(index=ES_BLOCK_INDEX)
            print("🔥 Elasticsearch client warmed up")
            return True
        except Exception as e:
            print(f"⚠️ ES client warm-up failed: {e}")
            return False
    return False


def get_es_client():
    """
    람다 환경에 최적화된 ES 클라이언트 관리
    - 컨테이너 재사용 시 기존 연결 활용
    - 연결 상태 확인 및 자동 재연결
    - 적절한 타임아웃 설정
    """
    global _es_client

    # 기존 클라이언트가 있으면 연결 상태 확인
    if _es_client is not None:
        try:
            # 간단한 ping으로 연결 상태 확인
            if _es_client.ping():
                print("\n✅ Elasticsearch client reused (healthy connection)")
                return _es_client
            else:
                print("⚠️ Elasticsearch connection unhealthy, reinitializing...")
                _es_client = None
        except Exception as e:
            print(f"⚠️ Elasticsearch connection check failed: {e}, reinitializing...")
            _es_client = None

    # 새 클라이언트 생성
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    password = os.getenv("ELASTIC_PASSWORD")

    if not cloud_id or not password:
        print("❌ Elasticsearch credentials not found")
        return None

    try:
        _es_client = Elasticsearch(
            cloud_id=cloud_id,
            basic_auth=("elastic", password),
            # 람다 환경에 최적화된 설정
            request_timeout=30,  # 30초 타임아웃
            max_retries=2,  # 최대 2회 재시도
            retry_on_timeout=True,
            # 연결 풀 설정 (람다에서는 작게)
            maxsize=2,
        )

        # 연결 테스트
        if _es_client.ping():
            # print("✅ Elasticsearch client initialized successfully")
            return _es_client
        else:
            print("❌ Elasticsearch ping failed after initialization")
            _es_client = None
            return None

    except Exception as e:
        print(f"❌ Failed to initialize Elasticsearch client: {e}")
        _es_client = None
        return None


def get_block_uri(
    stem_metadata: Dict[str, Any], root_s3_env_name="ROOT_BLOCK_OBJECT_URI"
) -> Optional[str]:
    """
    스템 오디오 URI 생성

    Args:
        stem_metadata (Dict[str, Any]): 스템 메타데이터

    Returns:
        str: 오디오 URL
    """
    try:
        songName = stem_metadata["id"]
        if songName[0] == "u":
            root_s3_env_name = "ROOT_USER_BLOCK_OBJECT_URI"
        root_s3 = os.getenv(root_s3_env_name)
        songId = stem_metadata["songId"].replace("#", "%23")
        sectionName = stem_metadata["sectionName"]
        stemType = stem_metadata["stemType"]
        path_uri = f"{root_s3}/{songId}/{sectionName}/{stemType}/{songName}"
        return path_uri
    except Exception as e:
        print(f"Error generating audio URI: {e}")
        return None


def get_context_song_stems(
    es_client,
    song_id: str,
    section_name: str,
    category: str = None,
    es_index=ES_BLOCK_INDEX,
) -> List[str]:
    """
    컨텍스트 곡 스템 조회
    """
    query = {
        "bool": {
            "must": [
                {"match": {"songId": song_id}},
                {"match": {"sectionName": section_name}},
            ],
            "must_not": [{"match": {"stemType": "mixed"}}],
        }
    }
    if category is not None:
        query["bool"]["must"].append({"match": {"stemType": category}})
    try:
        response = es_client.search(
            index=es_index,
            query=query,
            _source=SOURCE_LIST,
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        print(f"❌ Error getting context song stems: {e}")
        return []


def get_stem_metadata(
    es_client,
    index: str,
    stem_id: str,
    source: List[str] = SOURCE_LIST,
) -> Dict[str, Any]:
    """
    스템 메타데이터 조회

    Args:
        stem_id (str): 스템 ID

    Returns:
        Dict[str, Any]: 스템 메타데이터
    """
    return es_client.get(index=index, id=stem_id, _source=source)["_source"]


def create_es_query_filter(
    category: str,
    # context_song_info: Optional[Dict[str, Any]],
    context_song_info: Optional[ContextSong],
    target_music_info: Optional[Dict[str, Any]],
    continue_stem_info: Optional[Dict[str, Any]],
    request_type: Optional[str],
    remix_song_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    검색 필터 생성

    Args:
        category (str): 스템 카테고리
        context_song_info (Optional[Dict[str, Any]]): 컨텍스트 곡 정보

    Returns:
        Dict[str, Any]: 검색 필터 쿼리
    """

    # 필터 쿼리 구성
    common_filters = []

    if category == "mixed":
        remix_song_id = (
            remix_song_info.get("remix_song_id", None) if remix_song_info else None
        )
        remix_section_name = (
            remix_song_info.get("remix_section_name", None) if remix_song_info else None
        )
        if remix_section_name:
            common_filters.append({"match": {"sectionName": remix_section_name}})

        if remix_song_id:
            common_filters.append({"match": {"songId": remix_song_id}})
        else:
            if target_music_info:
                scale = target_music_info.get("scale")
                if scale:
                    if scale == "major":
                        common_filters.append({"wildcard": {"key": "*M"}})
                    elif scale == "minor":
                        common_filters.append({"wildcard": {"key": "*m"}})

                bpm_info = target_music_info.get("bpm")
                if bpm_info and isinstance(bpm_info, dict):
                    if "min" in bpm_info and "max" in bpm_info:
                        common_filters.append(
                            {
                                "range": {
                                    "bpm": {
                                        "gte": bpm_info["min"] - 10,
                                        "lte": bpm_info["max"] + 10,
                                    }
                                }
                            }
                        )

    # other categories
    else:
        key_info = None
        scale_info = None
        if context_song_info:
            key_info = context_song_info.key

        # scale_info = context_song_info.get("scale")
        if key_info and key_info not in ["", "-"]:
            if key_info[-1] == "M":
                scale_info = "major"
            elif key_info[-1] == "m":
                scale_info = "minor"

            candidate_keys, _ = get_candidate_key_difference(
                reference_key=key_info, isMixMajorMinor=False, keyRange=3
            )
            # candidate_keys를 사용해서 key 필드 필터링
            if candidate_keys and category not in ["fx", "rhythm"]:
                common_filters.append({"terms": {"key": candidate_keys}})

        if scale_info:
            if category not in ["fx"]:
                if scale_info == "major":
                    common_filters.append({"wildcard": {"key": "*M"}})
                elif scale_info == "minor":
                    common_filters.append({"wildcard": {"key": "*m"}})
        if continue_stem_info and "song_id" in continue_stem_info:
            print(f"continue_stem_info: {continue_stem_info}")
            if continue_stem_info["song_id"] and request_type == "continue":
                common_filters.append(
                    {"match": {"songId": continue_stem_info["song_id"]}}
                )
        # # REMIX_ONLY_TENANT 리스트에 포함된 모든 tenantName을 제외
        # bool_query["must_not"] = [{"terms": {"tenantName": REMIX_ONLY_TENANT}}]

    #! key & scale
    common_filters.append({"match": {"stemType": category}})
    common_filters.append({"match": {"isUnloopableOutro": False}})
    common_filters.append({"match": {"contractStatus": "active"}})

    # bool_query 구성
    bool_query = {}

    # filter 조건 추가
    if common_filters:
        bool_query["filter"] = common_filters

    # must_not 조건 추가 (REMIX_ONLY_TENANT 제외 - mixed 카테고리 제외)
    if category != "mixed":
        bool_query["must_not"] = [{"terms": {"tenantName": REMIX_ONLY_TENANT}}]
    filter_query = {"bool": bool_query}

    # return common_filters
    return filter_query


def get_song_info(
    es_client, song_id: str, es_index=ES_SONG_INDEX, source: List[str] = SOURCE_LIST
) -> Dict[str, Any]:
    """
    곡 정보 조회
    """
    return es_client.get(index=es_index, id=song_id, _source=source)["_source"]


def create_knn_retriever(
    field: str,
    query_vector: List[float],
    filter_query: Optional[Dict[str, Any]],
    k: int = 100,
    num_candidates: int = 300,
) -> Optional[Dict[str, Any]]:
    """KNN retriever 생성 헬퍼 함수"""
    # 필드별 예상 차원 정의
    expected_dimensions = {
        "musicEmbed": 256,
        "captionEmbed": 1536,
        "mtrppEmbed": 128,
        "chromaEmbed": 32,
    }

    # 차원 검증
    expected_dim = expected_dimensions.get(field)
    if expected_dim and len(query_vector) != expected_dim:
        return None

    knn_config = {
        "field": field,
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }

    # filter가 None이 아닐 때만 추가
    if filter_query is not None:
        knn_config["filter"] = filter_query

    return {"knn": knn_config}


def create_es_query(
    category: str,
    # context_song_info: Optional[Dict[str, Any]],
    context_song_info: Optional[ContextSong],
    target_music_info: Optional[Dict[str, Any]],
    text_embedding: List[float],
    music_text_joint_embedding: List[float],
    continue_stem_info: Optional[Dict[str, Any]],
    request_type: Optional[str],
    remix_song_info: Optional[Dict[str, Any]],
    # music_embedding: List[float],
) -> Optional[Dict[str, Any]]:

    # 필터 생성
    filter_query = create_es_query_filter(
        category=category,
        context_song_info=context_song_info,
        target_music_info=target_music_info,
        continue_stem_info=continue_stem_info,
        request_type=request_type,
        remix_song_info=remix_song_info,
    )

    retrievers = []
    embedding_configs = [
        ("captionEmbed", text_embedding, 50, 300),
        ("mtrppEmbed", music_text_joint_embedding, 50, 300),
        # ("musicEmbed", music_embedding, 50, 300),
    ]

    for field, embedding, k, num_candidates in embedding_configs:
        if embedding is not None and len(embedding) > 0:
            retriever = create_knn_retriever(
                field, embedding, filter_query, k, num_candidates
            )
            if retriever is not None:  # 차원 검증을 통과한 경우만 추가
                retrievers.append(retriever)

    # RRF 쿼리 구성
    if retrievers:
        es_query = {
            "retriever": {
                "rrf": {
                    "retrievers": retrievers,
                    "rank_window_size": 100,
                    "rank_constant": 10,
                }
            }
        }
    else:
        # retrievers가 비어있는 경우 기본 쿼리 사용
        es_query = {"match_all": {}}
        if filter_query:
            es_query = {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": (
                        filter_query["bool"]["filter"]
                        if filter_query.get("bool", {}).get("filter")
                        else []
                    ),
                }
            }

    return es_query


def search_music_stems_with_metadata(
    es_client,
    category: str,
    context_song_info: Optional[ContextSong],
    target_music_info: Optional[Dict[str, Any]],
    text_embedding: List[float],
    music_text_joint_embedding: List[float],
    continue_stem_info: Optional[Dict[str, Any]],
    request_type: Optional[str],
    remix_song_info: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    음악 스템 검색 (메타데이터 포함) - 더 효율적인 방식

    Returns:
        List[Dict]: [{"_source": metadata, "_score": float}, ...]
    """

    if es_client is None:
        raise Exception("Elasticsearch client not available")

    try:
        # 실제 처리 로직
        es_query = create_es_query(
            category=category,
            context_song_info=context_song_info,
            target_music_info=target_music_info,
            text_embedding=text_embedding,
            music_text_joint_embedding=music_text_joint_embedding,
            continue_stem_info=continue_stem_info,
            request_type=request_type,
            remix_song_info=remix_song_info,
        )

        # RRF 쿼리인지 확인하여 적절한 검색 방법 사용
        if es_query and "retriever" in es_query:
            # RRF 쿼리의 경우 - 모든 필요한 메타데이터를 한번에 가져옴
            es_index = ES_BLOCK_INDEX
            if remix_song_info and category == "mixed":
                es_index = ES_USER_BLOCK_INDEX
            output = es_client.search(
                index=es_index,
                retriever=es_query["retriever"],
                _source=SOURCE_LIST,  # 모든 필요한 필드
                size=100,
            )
        # 일반 쿼리의 경우
        else:
            output = es_client.search(
                index=ES_BLOCK_INDEX,
                query=es_query,
                _source=SOURCE_LIST,  # 모든 필요한 필드
                size=100,
            )

        # 메타데이터와 점수를 함께 반환
        return output["hits"]["hits"]

    except Exception as e:
        print(f"Error searching for music stem_type {category}: {str(e)}")
        print(traceback.format_exc())
        return []


def weighted_random_selection_from_hits(
    hits: List[Dict[str, Any]], selection_count: int = 1
) -> List[Dict[str, Any]]:
    """
    Elasticsearch hits에서 점수 기반 가중치로 랜덤 선택

    Args:
        hits: Elasticsearch hits 결과 [{"_source": {}, "_score": float}, ...]
        selection_count: 선택할 개수

    Returns:
        선택된 hits 리스트
    """
    import numpy as np

    if not hits:
        return []

    # 방법 1: Elasticsearch 점수 사용
    scores = np.array([hit.get("_score", 0) for hit in hits])

    # 점수가 0이거나 음수인 경우를 처리
    if np.all(scores <= 0):
        # 점수가 모두 0 이하면 순서 기반 가중치 사용 (높은 순위일수록 높은 가중치)
        weights = np.linspace(1.0, 0.1, len(hits))
    else:
        # 실제 점수를 가중치로 사용
        weights = scores
        # 음수 점수가 있는 경우 최소값을 0으로 만들어 정규화
        if np.min(weights) < 0:
            weights = weights - np.min(weights)

    # 가중치 정규화
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        # 모든 가중치가 0인 경우 균등 분포
        weights = np.ones(len(hits)) / len(hits)

    # 가중치 기반 랜덤 선택
    selected_indices = np.random.choice(
        len(hits), size=min(selection_count, len(hits)), replace=False, p=weights
    )

    return [hits[idx] for idx in selected_indices]


def get_embedding_for_category(embedding_list, category):
    if not embedding_list:
        return []
    for item in embedding_list:
        if isinstance(item, dict) and item.get("category") == category:
            return item.get("embedding", [])
    return []


def print_output(output):
    print("\n📊 Prompt stem info: \n", output.prompt_stem_info)
    print("\n📊 Context song info: \n", vars(output.context_song_info))


async def search_stems_from_es(
    text_embeddings: List[Dict[str, Any]] = [],
    music_text_joint_embeddings: List[Dict[str, Any]] = [],
    context_song_info: Optional[ContextSong] = None,
    target_music_info: Optional[Dict[str, Any]] = None,
    request_type: Optional[str] = None,
    continue_stem_info: Optional[Dict[str, Any]] = None,
    dialog_uuid: Optional[str] = None,
    remix_song_info: Optional[Dict[str, Any]] = None,
    target_context_section_index: Optional[int] = None,
    # mix_stem_diff: Optional[List[Stem]] = [],
) -> SearchStemOutput:
    es_client = get_es_client()
    if es_client is None:
        print("Elasticsearch client not available")
        return SearchStemOutput(retrieved_stems=[])

    # music_embedding = params.get("music_embedding", []) #TODO
    # continue_stem_info = continue_stem_info.get("continue_stem_info", None)

    if text_embeddings:
        categories = [item["category"] for item in text_embeddings]

    # 변수 초기화
    prompt_stem_info = []

    for idx, category in enumerate(categories):
        # * 1. 해당 스템 타입의 임베딩 데이터 추출
        # stem_music_embedding = get_embedding_for_category(music_embedding, category)
        text_embedding = get_embedding_for_category(text_embeddings, category)
        music_text_joint_embedding = get_embedding_for_category(
            music_text_joint_embeddings, category
        )

        # * 2. 해당 스템 타입 ES 검색
        if (
            category == "mixed"
            and context_song_info
            and context_song_info.created_sections_order
        ):
            continue

        search_results = search_music_stems_with_metadata(
            es_client,
            category=category,
            context_song_info=context_song_info,
            target_music_info=target_music_info,
            text_embedding=text_embedding,
            music_text_joint_embedding=music_text_joint_embedding,
            continue_stem_info=continue_stem_info,
            request_type=request_type,
            remix_song_info=remix_song_info,
        )

        if not search_results:
            print(f"⚠️ No search results found for category: {category}")
            continue

        if category == "mixed" and not (
            context_song_info and context_song_info.created_sections_order
        ):
            selected_blocks = weighted_random_selection_from_hits(
                search_results, selection_count=1
            )
            selected_block_meta = selected_blocks[0][
                "_source"
            ]  # 첫 번째 선택된 블록의 메타데이터

            context_audio_uris = []
            context_song_stems = get_context_song_stems(
                es_client,
                selected_block_meta["songId"],
                selected_block_meta["sectionName"],
                es_index=ES_USER_BLOCK_INDEX if remix_song_info else ES_BLOCK_INDEX,
            )
            for stem_meta in context_song_stems:
                context_audio_uris.append(
                    get_block_uri(
                        stem_meta,
                        root_s3_env_name=(
                            "ROOT_BLOCK_OBJECT_URI"
                            if not remix_song_info
                            else "ROOT_USER_BLOCK_OBJECT_URI"
                        ),
                    )
                )

            song_info = get_song_info(
                es_client,
                selected_block_meta["songId"],
                es_index=ES_USER_SONG_INDEX if remix_song_info else ES_SONG_INDEX,
                source=["songStructure"],
            )

            context_song_info = ContextSong(
                song_id=selected_block_meta["songId"],
                bpm=selected_block_meta["bpm"],
                key=selected_block_meta["key"],
                bar_count=selected_block_meta["barCount"],
                section_name=selected_block_meta["sectionName"],
                section_role=selected_block_meta["sectionRole"],
                song_structure=song_info["songStructure"],
                context_audio_uris=context_audio_uris,
                created_sections_order=[
                    {
                        selected_block_meta["sectionName"]: selected_block_meta[
                            "sectionRole"
                        ]
                    }
                ],
                arranged_sections_order=[
                    {
                        selected_block_meta["sectionName"]: selected_block_meta[
                            "sectionRole"
                        ]
                    }
                ],
                is_remix=True if remix_song_info else False,
            )

        else:
            if target_context_section_index is not None:
                section_name, section_role = next(
                    iter(
                        context_song_info.arranged_sections_order[
                            target_context_section_index
                        ].items()
                    )
                )

                context_audio_uris = []
                context_song_stems = get_context_song_stems(
                    es_client,
                    context_song_info.song_id,
                    section_name,
                    es_index=ES_USER_BLOCK_INDEX if remix_song_info else ES_BLOCK_INDEX,
                )

                for idx, stem_meta in enumerate(context_song_stems):
                    if idx == 0:
                        context_song_info.bar_count = stem_meta["barCount"]
                        context_song_info.section_name = section_name
                        context_song_info.section_role = section_role
                    context_audio_uris.append(
                        get_block_uri(
                            stem_meta,
                            root_s3_env_name=(
                                "ROOT_BLOCK_OBJECT_URI"
                                if not remix_song_info
                                else "ROOT_USER_BLOCK_OBJECT_URI"
                            ),
                        )
                    )
                context_song_info.context_audio_uris = context_audio_uris

            selected_blocks = weighted_random_selection_from_hits(
                search_results, selection_count=4
            )

            for block_hit in selected_blocks:
                block_meta = block_hit["_source"]  # 메타데이터 추출

                block_meta["caption"] = (
                    f"{block_meta['caption'][: len(block_meta['caption']) // 2]} ..."
                )
                block_meta["uri"] = get_block_uri(block_meta) + ".aac"
                block_meta["dialog_uuid"] = dialog_uuid
                # TODO: use Stem instance instead of dict
                prompt_stem_info.append(block_meta)

    if isinstance(context_song_info, dict):
        context_song = ContextSong(
            song_id=context_song_info.get("songId"),
            bpm=context_song_info.get("bpm"),
            key=context_song_info.get("key"),
            bar_count=context_song_info.get("barCount"),
            section_name=context_song_info.get("sectionName"),
            section_role=context_song_info.get("sectionRole"),
            context_audio_uris=context_song_info.get("contextAudioUris"),
            created_sections_order=context_song_info.get("createdSectionsOrder"),
            arranged_sections_order=context_song_info.get("arrangedSectionsOrder"),
            is_remix=context_song_info.get("isRemix"),
        )
    else:
        context_song = context_song_info

    vocal_block = {}
    if context_song is None:
        raise Exception("Context song is None")
    if context_song.is_remix:
        vocal_block = get_context_song_stems(
            es_client,
            context_song.song_id,
            context_song.section_name,
            category="melody",
            es_index=ES_USER_BLOCK_INDEX,
        )
        if vocal_block:
            vocal_block = vocal_block[0]
            vocal_block["caption"] = (
                f"{vocal_block['caption'][: len(vocal_block['caption']) // 2]} ..."
            )
            vocal_block["uri"] = get_block_uri(
                vocal_block, "ROOT_USER_BLOCK_OBJECT_URI"
            )
            vocal_block["category"] = vocal_block.pop("stemType")
            vocal_block["isOriginal"] = True
            vocal_block["isBlock"] = True
            vocal_block["instrumentName"] = "vocal"
            vocal_block["dialog_uuid"] = dialog_uuid

    output = SearchStemOutput(
        prompt_stem_info=prompt_stem_info,
        context_song_info=context_song,
        vocal_stem_info=vocal_block,
    )
    print_output(output)
    return output


# %%
