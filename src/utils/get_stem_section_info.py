import os
import pathlib
from typing import Any, Dict, List, Optional

import anthropic
import openai
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field

from models import ContextSong, Stem

env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


ES_BLOCK_INDEX = os.getenv("ES_BLOCK_INDEX") or ""
ES_USER_BLOCK_INDEX = os.getenv("ES_USER_BLOCK_INDEX") or ""
ES_USER_SONG_INDEX = os.getenv("ES_USER_SONG_INDEX") or ""
ES_SONG_INDEX = os.getenv("ES_SONG_INDEX") or ""
ES_ENVIRONMENTAL_SOUND_INDEX = os.getenv("ES_ENVIRONMENTAL_SOUND_INDEX") or ""

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

_es_client = None


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


# def get_context_song_stems(
#     es_client, song_id: str, section_name: str, source: List[str] = SOURCE_LIST
# ) -> List[str]:
#     """
#     컨텍스트 곡 스템 조회
#     """
#     query = {
#         "bool": {
#             "must": [
#                 {"match": {"songId": song_id}},
#                 {"match": {"sectionName": section_name}},
#             ],
#             "must_not": [{"match": {"stemType": "mixed"}}],
#         }
#     }

#     try:
#         response = es_client.search(
#             index=ES_BLOCK_INDEX,
#             query=query,
#             _source=SOURCE_LIST,
#         )
#         return [hit["_source"] for hit in response["hits"]["hits"]]
#     except Exception as e:
#         print(f"❌ Error getting context song stems: {e}")
#         return []


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
        root_s3 = os.getenv(root_s3_env_name)
        songId = stem_metadata["songId"].replace("#", "%23")
        sectionName = stem_metadata["sectionName"]
        stemType = stem_metadata["stemType"]
        songName = stem_metadata["id"]
        path_uri = f"{root_s3}/{songId}/{sectionName}/{stemType}/{songName}"

        return path_uri
    except Exception as e:
        print(f"Error generating audio URI: {e}")
        return None


def update_mix_stems(
    chosen_sections,
    selected_stem_diff,
    is_new_section,
    working_section_index,
    request_type,
):
    """
    중첩된 리스트를 주어진 조건에 따라 업데이트합니다.

    Args:
        mix_stems (list): 업데이트할 대상 리스트
        stem_info (dict): 추가할 딕셔너리
        target_index (int): 아이템을 추가할 위치.
    """
    print("before print(chosen_sections)")
    print(chosen_sections)
    print("working_section_index: ", working_section_index)
    if isinstance(selected_stem_diff, dict):
        selected_stem = Stem(
            id=selected_stem_diff.get("id", None),
            mix_id=selected_stem_diff.get("mixId", None),
            dialog_uuid=selected_stem_diff.get("dialogUuid", None),
            is_original=selected_stem_diff.get("isOriginal", None),
            is_block=selected_stem_diff.get("isBlock", None),
            category=selected_stem_diff.get("category", None),
            caption=selected_stem_diff.get("caption", None),
            instrument_name=selected_stem_diff.get("instrumentName", None),
            section_name=selected_stem_diff.get("sectionName", None),
            section_role=selected_stem_diff.get("sectionRole", None),
            bar_count=selected_stem_diff.get("barCount", None),
            bpm=selected_stem_diff.get("bpm", None),
            key=selected_stem_diff.get("key", None),
            uri=selected_stem_diff.get("uri", None),
            url=selected_stem_diff.get("url", None),
        )
    else:
        selected_stem = selected_stem_diff
    # if request_type in ["replace", "remove"]:
    selected_dialog_uuid = selected_stem.dialog_uuid

    # dialog_uuid로 기존 스템 제거 (working_section_index != -1인 경우만)
    if (
        selected_dialog_uuid
        and chosen_sections != []
        and not (is_new_section and working_section_index == 0)
    ):
        for idx, stem in enumerate(chosen_sections[working_section_index]):
            if stem.dialog_uuid == selected_dialog_uuid:
                chosen_sections[working_section_index].pop(idx)
                break

    # if selected_dialog_uuid and chosen_sections != []:
    #     for idx, stem in enumerate(chosen_sections[working_section_index]):
    #         if stem.dialog_uuid == selected_dialog_uuid:
    #             chosen_sections[working_section_index].pop(idx)
    #             break

    if request_type != "remove":
        # if working_section_index == 0 and is_new_section:
        if is_new_section:
            if selected_stem_diff:
                # chosen_sections.insert(working_section_index, [selected_stem])
                chosen_sections[working_section_index].append(selected_stem)
            else:
                chosen_sections.insert(working_section_index, [])
                print("check add empty list")
                print(chosen_sections)
            return chosen_sections
        else:
            # index가 리스트의 유효한 범위 내에 있는지 확인
            # 있다면 해당 sub-list에 아이템을 append
            if 0 <= working_section_index < len(chosen_sections):
                if selected_stem_diff:
                    chosen_sections[working_section_index].append(selected_stem)
                else:
                    chosen_sections.insert(working_section_index, [])
            # index가 범위를 벗어나면 (새로운 sub-list를 만들어야 할 때)
            # 해당 위치에 [item]을 insert
            else:
                if selected_stem_diff:
                    chosen_sections.insert(working_section_index, [selected_stem])
                else:
                    chosen_sections.insert(working_section_index, [])
        print("Chosen sections after update:")
        print(chosen_sections)
    return chosen_sections


# %% SECTION INFO


# 입력/출력 스키마 정의
class SectionPlannerInput(BaseModel):
    """섹션 계획 도구에 대한 입력 스키마"""

    user_request: Optional[str] = Field(
        default="", description="사용자의 특별한 요청 (없으면 자동 진행)"
    )
    context_song_info: Dict[str, Any] = Field(..., description="현재 곡 정보")
    mix_stem_diff: List[Dict[str, Any]] = Field(
        default=[], description="현재 믹스에 포함된 스템 정보"
    )


class SectionPlannerOutput(BaseModel):
    """섹션 계획 도구에 대한 출력 스키마"""

    nextSectionName: str = Field(..., description="다음 섹션 이름 (A-P)")
    nextSectionRole: str = Field(..., description="다음 섹션 역할")
    nextSectionIndex: int = Field(..., description="다음 섹션 인덱스")
    # next_section_bar_count: int = Field(..., description="다음 섹션 바 개수")
    createdSectionsOrder: List[Dict[str, str]] = Field(
        ...,
        description="Section name as key and role as value in chronological order",
    )
    arrangedSectionsOrder: List[Dict[str, str]] = Field(
        ...,
        description="Section name as key and role as value in musical order",
    )

    reasoning: str = Field(..., description="결정 과정에 대한 설명")


# LLM 설정
CLAUDE_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 1000
TEMPERATURE = 0.6
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


# 도구 스키마 정의
section_planner_tool_schema = {
    "name": "plan_next_section",
    "description": "Analyze and determine the next section to generate",
    "input_schema": {
        "type": "object",
        "properties": {
            "nextSectionName": {
                "type": "string",
                "description": "Next section name (A, B, C, D, E, F, G, H)",
            },
            "nextSectionRole": {
                "type": "string",
                "description": "Next section role (e.g. Verse, Chorus / Drop, Outro)",
            },
            "nextSectionIndex": {
                "type": "integer",
                "description": "Next section index (position in the musical structure)",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of why this section was selected and the decision-making process",
            },
        },
        "required": [
            "nextSectionName",
            "nextSectionRole",
            "nextSectionIndex",
            "reasoning",
        ],
    },
}


async def call_claude_api_for_section(system_message, messages):
    """Claude API 호출 함수"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=CLAUDE_MODEL,
        system=system_message,
        messages=messages,
        tools=[section_planner_tool_schema],
        tool_choice={"type": "tool", "name": "plan_next_section"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


async def call_openai_api_for_section(system_message, messages):
    """OpenAI API 호출 함수"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # OpenAI 함수 호출 형식으로 변환
    openai_function = {
        "name": section_planner_tool_schema["name"],
        "description": section_planner_tool_schema["description"],
        "parameters": section_planner_tool_schema["input_schema"],
    }

    # OpenAI 메시지 형식으로 변환
    openai_messages = []

    # system_message 처리
    if isinstance(system_message, list):
        system_content = "\n".join(
            [msg["text"] for msg in system_message if msg.get("type") == "text"]
        )
    else:
        system_content = system_message

    openai_messages.append({"role": "system", "content": system_content})
    openai_messages.append({"role": "user", "content": messages})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=openai_messages,
        functions=[openai_function],
        function_call={"name": "plan_next_section"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


async def call_llm_for_section_planning(
    system_message, messages, llm_provider=DEFAULT_LLM_PROVIDER
):
    """LLM 호출 통합 함수"""
    print("\n✅ Section planning: call_llm_for_section_planning")
    if llm_provider.lower() == "openai":
        # print("\n✅ Section planning: call_openai_api")
        return await call_openai_api_for_section(system_message, messages)
    elif llm_provider.lower() == "claude":
        # print("\n✅ Section planning: call_claude_api")
        return await call_claude_api_for_section(system_message, messages)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


async def get_next_section_info_llm(
    context_song_info: ContextSong, mix_stem_diff, user_request=""
):
    """
    Determine the next section to generate based on the current section information.

    Args:
        context_song_info: Context song info dictionary
        mix_stem_diff: Current mix stems info
        user_request: User request

    Returns:
        dict: Next section info
        - nextSectionName: Next section name (A, B, C, D ,...)
        - nextSectionRole: Next section role
        - nextSectionIndex: Next section index
        - createdSectionsOrder: Created sections list in order
        - arrangedSectionsOrder: Arranged sections list in order
        - reasoning: Reasoning process
    """

    system_message = build_system_message()

    # 섹션과 역할 정보를 함께 생성
    songStructure = context_song_info.song_structure
    created_sections_order = context_song_info.created_sections_order
    arranged_sections_order = context_song_info.arranged_sections_order

    # 섹션들을 간단한 딕셔너리 형태로 변환
    def convert_to_simple_dicts(sections):
        """섹션 데이터를 {sectionName: role} 딕셔너리 형태로 변환"""
        if not sections:
            return []

        if isinstance(sections[0], dict):
            # 이미 딕셔너리 형태인 경우
            if "sectionName" in sections[0] and "sectionRole" in sections[0]:
                # {sectionName: name, sectionRole: role} → {name: role} 형태로 변환
                return [
                    {section["sectionName"]: section["sectionRole"]}
                    for section in sections
                ]
            else:
                # 이미 {name: role} 형태인 경우
                return sections
        elif isinstance(sections[0], (tuple, list)) and len(sections[0]) == 2:
            # tuple/list 형태인 경우 (name, role)
            return [{section[0]: section[1]} for section in sections]
        else:
            # 문자열 배열인 경우
            return [
                {section: songStructure.get(section, "Unknown")} for section in sections
            ]

    created_sections_dicts = convert_to_simple_dicts(created_sections_order)
    arranged_sections_dicts = convert_to_simple_dicts(arranged_sections_order)

    # 사용된 역할들 분석 (간단한 딕셔너리 형태에서)
    used_roles = []
    role_counts = {}

    for section_dict in created_sections_dicts:
        for section_name, role in section_dict.items():
            if role not in used_roles:
                used_roles.append(role)
            role_counts[role] = role_counts.get(role, 0) + 1

    # 현재 섹션의 arranged_sections에서의 위치 찾기
    current_section_name = context_song_info.section_name
    current_position_in_arranged = -1
    for i, section_dict in enumerate(arranged_sections_dicts):
        if current_section_name in section_dict:
            current_position_in_arranged = i
            break

    messages = f"""Analyze the current song status and determine the next section following the system instructions:

**User request**: {user_request if user_request else "No special request (automatic progress)"}

**Current song data**:
- Current section: {context_song_info.section_name} ({context_song_info.section_role})
- Current working section position in arranged order(workingSectionIndex): {current_position_in_arranged}

**Reference song structure (songStructure)**:
{songStructure}

**Creation status**:
- createdSectionsOrder (chronological order): {created_sections_dicts}
- arrangedSectionsOrder (musical order): {arranged_sections_dicts}
- Used roles: {used_roles}
- Role counts: {role_counts}

**Current mix stems**: {[stem.category for stem in mix_stem_diff]}

Follow the Required Analysis Process outlined in the system message and provide detailed reasoning."""

    response = await call_llm_for_section_planning(system_message, messages)

    # OpenAI 응답 처리
    if hasattr(response, "choices") and response.choices:
        if response.choices[0].message.function_call:
            import json

            function_args = json.loads(
                response.choices[0].message.function_call.arguments
            )

            # 현재 생성된 섹션들 + 새로운 섹션 추가
            new_section = {
                function_args["nextSectionName"]: function_args["nextSectionRole"]
            }
            function_args["createdSectionsOrder"] = created_sections_dicts + [
                new_section
            ]

            insert_index = function_args.get("nextSectionIndex", 0)
            print(f"💬 Insert index: {insert_index}")

            new_arranged = arranged_sections_dicts.copy()
            new_arranged.insert(insert_index, new_section)
            function_args["arrangedSectionsOrder"] = new_arranged

            result = SectionPlannerOutput(**function_args)

    else:
        # 기본 응답
        raise ValueError("LLM 응답을 처리할 수 없습니다.")

    es_client = get_es_client()
    stems_info = get_context_song_stems(
        es_client,
        context_song_info.song_id,
        result.nextSectionName,
        es_index=ES_USER_BLOCK_INDEX if context_song_info.is_remix else ES_BLOCK_INDEX,
    )
    print(stems_info)
    if len(stems_info) == 0:
        raise Exception("No stems found")
    bar_count = stems_info[0].get("barCount", None)

    new_context_song = ContextSong(
        song_id=context_song_info.song_id,
        bpm=context_song_info.bpm,
        key=context_song_info.key,
        bar_count=bar_count,
        section_name=result.nextSectionName,
        song_structure=context_song_info.song_structure,
        section_role=result.nextSectionRole,
        context_audio_uris=[
            get_block_uri(
                stem_info,
                root_s3_env_name=(
                    "ROOT_BLOCK_OBJECT_URI"
                    if not context_song_info.is_remix
                    else "ROOT_USER_BLOCK_OBJECT_URI"
                ),
            )
            for stem_info in stems_info
        ],
        created_sections_order=result.createdSectionsOrder,
        arranged_sections_order=result.arrangedSectionsOrder,
        is_remix=context_song_info.is_remix,
    )
    print("new_context_song")
    print(new_context_song.to_dict())

    return {
        "context_song": new_context_song,
        "nextSectionIndex": result.nextSectionIndex,
        "reasoning": result.reasoning,
    }


def build_system_message():
    return """
[1. Role and Goal]
You are a professional music producer who analyzes the structure of a song and determines the next section to generate using progressive expansion approach based on adjacent sections.

[2. Data Structure Understanding]
- **createdSectionsOrder**: Chronological list of created sections [{sectionName: role}, ...]
- **arrangedSectionsOrder**: Musical order list from song start to end [{sectionName: role}, ...]
- **songStructure**: Reference song structure {sectionName: role, ...}. Follow the sequential order as reference, but skip consecutive sections with the same role to maintain variety.
- **workingSectionIndex**: User's current working position in arrangedSectionsOrder
- **sectionName**: Current section being worked on

[3. Analysis Process]

**Step 1: Situational Analysis**
- Locate current section position in songStructure and arrangedSectionsOrder
- Please notice that the first section of arrangedSectionsOrder or createdSectionsOrder can start from the middle of the music based on songStructure .
- Analyze used roles and calculate role counts from createdSectionsOrder. 
- Evaluate musical completion status and structural requirements

**Step 2: Direction and Candidate Selection**
- **Direction logic**: 
  * **MUSICAL STRUCTURE PRIORITY**: Consider logical musical flow first
  * **INTRO SPECIAL RULE**: If selecting Intro role, ALWAYS place at beginning (nextSectionIndex = 0) regardless of user request, unless user explicitly specifies a different position
  * **OUTRO SPECIAL RULE**: If selecting Outro role, ALWAYS place at end (nextSectionIndex = length of arrangedSectionsOrder)
  * **Forward direction**: Default for most cases, following songStructure order
  * **Backward direction**: Only when inserting sections that logically come BEFORE current position
  * **CRITICAL**: Pre-Chorus/Build-Up should NEVER come before Intro
  * **CRITICAL**: Bridge/Breakdown should typically come after Verse/Chorus, not before
- **Candidate search**: Search adjacent sections in distance order from current position in songStructure
- **Role continuity check**: If next adjacent section has the same role as current section, skip to find different role section
- **Validation criteria** (priority order):
  * **Musical logic**: Section placement must make musical sense (e.g., no Pre-Chorus before Intro)
  * Section not yet in createdSectionsOrder
  * Role variety: Prefer different roles over consecutive same roles
  * Role diversity priority: unused roles > roles with count 
  * Maximum 2 sections per role (unless user specifically requests more)
  * Give weight to selecting the closest candidate among the valid candidates, but it is not necessary to select the closest one depending on the conditions.

**Step 3: Index Calculation and Updates**
- **Index determination (CRITICAL)**:
  * **STEP 3.1 - CHECK ROLE FIRST**: 
    - If nextSectionRole = "Intro": IMMEDIATELY set nextSectionIndex = 0 and skip to array updates
    - If nextSectionRole = "Outro": IMMEDIATELY set nextSectionIndex = length of arrangedSectionsOrder and skip to array updates
  * **STEP 3.2 - For non-Intro/Outro roles only**: Evaluate user requests and direction
    - User-specific requests: 
      * "after current section" = workingSectionIndex + 1
      * "before current section" = workingSectionIndex
      * "insert at beginning" = 0
    - Automatic mode:
      * Backward direction (inserting at beginning) = 0
      * Forward direction = current_section_position + 1
- **Array updates**:
  * createdSectionsOrder: append {new_section_name: new_section_role}
  * arrangedSectionsOrder: insert new section at calculated index position

[4. Index Calculation Examples]
- **Role-based priority examples**:
  * Selecting "Intro" role → nextSectionIndex = 0 (ALWAYS at beginning)
  * Selecting "Outro" role → nextSectionIndex = length of arrangedSectionsOrder (ALWAYS at end)

- **User-directed insertions (non-Intro/Outro)**:
  * "after current section" (workingSectionIndex=0) → nextSectionIndex = 1
  * "before current section" (workingSectionIndex=2) → nextSectionIndex = 2

- **Automatic mode (non-Intro/Outro)**:
  * Backward direction (insert at beginning) → nextSectionIndex = 0
  * Forward direction (current at position 1) → nextSectionIndex = 2

[5. Core Rules]
1. **INTRO/OUTRO PLACEMENT PRIORITY**: 
   - Intro ALWAYS goes to beginning (nextSectionIndex = 0) 
   - Outro ALWAYS goes to end (nextSectionIndex = length of arrangedSectionsOrder)
   - These rules override all other placement logic including user requests
2. **MUSICAL LOGIC FIRST**: Section placement must follow logical musical structure (Intro → [Verse → Pre-Chorus/Build-up → Chorus/Drop → Bridge/Breakdown] x N → Outro etc.). 
   - '[ ]' part may be repeated 'N' times depending on the song structure.
3. **Musical Context Priority**: Consider overall song structure and completion requirements
4. **Distance-Based Selection**: Search and choose closest valid adjacent section
5. **Role Variety**: Avoid consecutive sections with same role, skip to different role when possible
6. **Role Diversity**: Prioritize unused roles, enforce duplication limits (max 2 per role)
7. **Structure Respect**: Maintain alphabetical order consistency with songStructure
8. **Position Awareness**: Consider workingSectionIndex for user-directed insertions (only for non-Intro/Outro)
9. **Musical logic examples**:
- Current: Intro, Next: Bridge → REJECT: Bridge should come after Verse/Chorus/Drop
- Current: Verse, Next: Pre-Chorus → ACCEPT: Pre-Chorus can come after Verse
- Current: Chorus, Next: Bridge → ACCEPT: Bridge can come after Chorus
- Selecting Intro role → ALWAYS nextSectionIndex = 0 (beginning)
- Selecting Outro role → ALWAYS nextSectionIndex = arrangedSectionsOrder length (end)

Follow this process systematically and provide detailed step-by-step reasoning for each decision.
"""


# %%
