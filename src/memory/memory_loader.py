# modules/memory_loader.py

import json
import logging
import os
import pprint
from typing import List, Optional

from common.cache import cache_get
from models import ContextSong, MemoryData, Stem
from utils.get_stem_section_info import get_next_section_info_llm

logger = logging.getLogger(__name__)


is_local = os.getenv("WEBSOCKET_API_ENDPOINT") is None

# Configure logging with handler - set root to INFO to reduce noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console
)

# Enable DEBUG for our internal modules only
if is_local:
    # Load environment variables if run in local
    # Enable DEBUG for main module only - suppress operational logs
    logger.setLevel(logging.DEBUG)
    # Suppress operational INFO logs from internal modules
    logging.getLogger("utils.data_schema_mapper").setLevel(logging.WARNING)
    logging.getLogger("tools.mcp_base").setLevel(logging.WARNING)

else:
    logger.setLevel(logging.INFO)


def _is_valid_schema(schema: dict) -> bool:
    return schema["request"] and any(
        ans for ans in schema["request"][0]["answers"] if ans["status"] != "error"
    )


def _load_schema(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_latest_valid_memory(folder: str) -> Optional[dict]:

    files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.startswith("data_schema_") and f.endswith(".json")
        ],
        key=lambda x: os.path.getmtime(os.path.join(folder, x)),
        reverse=True,
    )

    for file in files:
        data = _load_schema(os.path.join(folder, file))
        if data and _is_valid_schema(data):
            return _extract_metadata(os.path.join(folder, file), data)
    return None


def load_top_n_valid_memories(
    folder: str, candidate_memory_count: int
) -> List[MemoryData]:
    files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.startswith("data_schema_") and f.endswith(".json")
        ],
        key=lambda x: os.path.getmtime(os.path.join(folder, x)),
        reverse=True,
    )
    valid_memories = []
    for file in files:
        if len(valid_memories) >= candidate_memory_count:
            break
        data = _load_schema(os.path.join(folder, file))
        if data and _is_valid_schema(data):
            valid_memories.append(_extract_metadata(os.path.join(folder, file), data))
    return valid_memories


def get_recent_valid_dialog_uuids(
    current_dialog_uuid: str, session_uuid: str, max_count: int = 10
) -> List[str]:
    logger.debug(f"session_uuid: {session_uuid}")
    dialog_uuids_in_session = cache_get(f"dialogs_in_session:{session_uuid}")
    logger.debug(f"dialog_uuids_in_session: {dialog_uuids_in_session}")
    if dialog_uuids_in_session:
        dialog_uuids_in_session = json.loads(dialog_uuids_in_session)
    else:
        dialog_uuids_in_session = []

    valid_dialogs = []

    # reverse chronological order
    for dialog_uuid in dialog_uuids_in_session[::-1]:
        dialog = cache_get(f"request_answer:{dialog_uuid}")
        if dialog:
            dialog = json.loads(dialog)
        else:
            continue
        if (
            dialog_uuid != current_dialog_uuid
            and has_answer(dialog)
            and no_errors(dialog)
        ):
            valid_dialogs.append(dialog_uuid)
            if len(valid_dialogs) >= max_count:
                break
    return valid_dialogs


def has_answer(dialog: dict) -> bool:
    answers = dialog.get("answers", [])
    return answers and len(answers) > 0


def no_errors(dialog: dict) -> bool:
    answers = dialog.get("answers", [])
    return all(ans.get("status") != "error" for ans in answers)


def _extract_metadata(path: str, data: dict) -> MemoryData:
    req = data["request"][0]
    ans = next(ans for ans in req["answers"] if ans["status"] != "error")

    memory_data = MemoryData(
        memory_id=path,
        user_prompt=req["chatMessage"]["chatText"],
        intent_focused_prompt=req["chatMessage"].get("intentFocusedPrompt", ""),
        intent_history=req.get("context", {}).get("previousContext", []),
        chosen_sections=ans.get("mix", {}).get("mixData", {}).get("stems", []),
        generated_stems=ans.get("suggestedStems", []),
        context_song=req["requestInformation"].get("contextSongInfo", {}),
        working_section_index=req["requestInformation"].get("workingSectionIndex", 0),
    )

    return memory_data


async def extract_data_from_memory(
    user_prompt, dialog_uuid, session_uuid, is_local=True
):
    from memory.orchestrator import select_best_memory

    # TODO: rework this function / function name
    memory_data, strategy, llm_usage = select_best_memory(
        user_prompt, dialog_uuid, session_uuid, is_local
    )

    logger.debug("\n🎯 Strategy:\n %s", strategy)

    previous_working_section_index = memory_data.working_section_index
    if strategy.is_publish_song:
        logger.debug("\n🎯 Publishing song: %s", dialog_uuid)
        return memory_data, strategy, llm_usage, 0
    else:
        if strategy.is_start_from_scratch:
            logger.debug("\n🎯 Starting fresh, creating new task_id:\n %s", dialog_uuid)
            return MemoryData(), strategy, llm_usage, 0
        else:
            if (
                strategy.should_load_older_memories
                and len(strategy.selected_older_memory_ids) > 0
            ):
                #! Case 3: 직전 상태 읽어서, 생성 정보는 이전 메모리 정보 읽어서.
                logger.debug(
                    "\n🎯 Selected memory decision:\n %s",
                    strategy.selected_older_memory_ids,
                )
                selected_memory_id = strategy.selected_older_memory_ids[0]
                selected_memory_data = load_memory_from_cache(selected_memory_id)
                if strategy.is_use_suggested_stems:
                    if isinstance(selected_memory_data.generated_stems[0], dict):
                        memory_data.generated_stems = [
                            Stem(
                                id=stem.get("id", ""),
                                mix_id=stem.get("mixId", ""),
                                dialog_uuid=stem.get("dialogUuid", ""),
                                is_original=stem.get("isOriginal", ""),
                                is_block=stem.get("isBlock", ""),
                                category=stem.get("category", ""),
                                caption=stem.get("caption", ""),
                                instrument_name=stem.get("instrumentName", ""),
                                section_name=stem.get("sectionName", ""),
                                section_role=stem.get("sectionRole", ""),
                                bar_count=stem.get("barCount", ""),
                                bpm=stem.get("bpm", ""),
                                key=stem.get("key", ""),
                                uri=stem.get("uri", ""),
                                url=stem.get("url", ""),
                            )
                            for stem in selected_memory_data.generated_stems
                        ]

                    else:
                        memory_data.generated_stems = (
                            selected_memory_data.generated_stems
                        )
                    memory_data.working_section_index = (
                        selected_memory_data.working_section_index
                    )
                else:
                    memory_data.generated_stems = []

                #! Case 4: 특정 시점 부터 새로운 브랜치 생성
                if strategy.is_start_new_branch:
                    logger.debug(
                        "\n🎯 Starting new branch, creating new task_id:\n %s",
                        dialog_uuid,
                    )
                    memory_data.context_song = selected_memory_data.context_song
                    memory_data.intent_history = selected_memory_data.intent_history
                    memory_data.chosen_sections = selected_memory_data.chosen_sections
                    memory_data.generated_stems = selected_memory_data.generated_stems
                    memory_data.turn_index = selected_memory_data.turn_index
                    memory_data.working_section_index = (
                        selected_memory_data.working_section_index
                    )

            else:
                #! Case 2: 이전 요청 이어서 생성
                logger.debug(f"\n🎯 Selected recent task_id: {memory_data.memory_id}")
                if not strategy.is_use_suggested_stems:
                    memory_data.generated_stems = []
                # # TODO: how to get context song? -> 특정 섹션을 언급한거면 해당 context song info가 필요하다. 지금은 현재 정보가 저장되어 있으니까.

                #! Case 5: 새로운 섹션 시작
                if strategy.is_start_new_section:
                    working_section = []
                    if (
                        memory_data.working_section_index
                        >= len(memory_data.chosen_sections)
                        or memory_data.working_section_index == -1
                    ):
                        working_section = []
                    else:
                        working_section = memory_data.chosen_sections[
                            memory_data.working_section_index
                        ]

                    logger.debug(
                        "\n🎯 Starting new section, creating new task_id: %s",
                        dialog_uuid,
                    )

                    next_section_info = await get_next_section_info_llm(
                        memory_data.context_song,
                        working_section,
                        strategy.intent_focused_prompt,
                    )
                    llm_usage += 1

                    logger.debug(
                        "\n🎯 Next section info: %s",
                        pprint.pformat(next_section_info),
                    )

                    memory_data.context_song = next_section_info["context_song"]
                    memory_data.working_section_index = next_section_info[
                        "nextSectionIndex"
                    ]
                    memory_data.working_section = working_section

            logger.debug("\n\n=== === Current Info === ===")
            logger.debug("\n💬 User prompt:\n %s", user_prompt)
            logger.debug(
                "\n💬 Current Mix index:\n %s", memory_data.working_section_index
            )
            logger.debug(
                "\n💬 Current Mix stem diff:\n %s", memory_data.working_section
            )

            logger.debug("\n\n=== === Previous Context === ===")
            logger.debug("💬 Context song info:\n %s", memory_data.context_song)
            logger.debug("\n💬 Previous context:\n %s", memory_data.intent_history)
            logger.debug("\n💬 Generated stem diff:\n %s", memory_data.generated_stems)
            logger.debug("\n💬 Mix stem diff:\n %s", memory_data.working_section)
            logger.debug("=================================\n")

            return memory_data, strategy, llm_usage, previous_working_section_index


def memory_from_data_schema(data_schema: dict) -> MemoryData:

    # context_song_dict = data_schema.context_song
    context_song_dict = data_schema.context_song or {}

    context_song = ContextSong(
        song_id=context_song_dict.get("songId", ""),
        bpm=context_song_dict.get("bpm", 0),
        key=context_song_dict.get("key", ""),
        bar_count=context_song_dict.get("barCount", 0),
        section_name=context_song_dict.get("sectionName", ""),
        song_structure=context_song_dict.get("songStructure", ""),
        section_role=context_song_dict.get("sectionRole", ""),
        context_audio_uris=context_song_dict.get("contextAudioUris", []),
        created_sections_order=context_song_dict.get("createdSectionsOrder", []),
        arranged_sections_order=context_song_dict.get("arrangedSectionsOrder", []),
        is_remix=context_song_dict.get("isRemix", False),
    )

    return MemoryData(
        memory_id=data_schema.memory_id,
        user_prompt=data_schema.user_prompt,
        intent_focused_prompt=data_schema.intent_focused_prompt,
        intent_history=data_schema.intent_history,
        chosen_sections=[
            [
                Stem(
                    id=stem.get("id", ""),
                    mix_id=stem.get("mixId", ""),
                    dialog_uuid=stem.get("dialogUuid", ""),
                    is_original=stem.get("isOriginal", ""),
                    is_block=stem.get("isBlock", ""),
                    category=stem.get("category", ""),
                    caption=stem.get("caption", ""),
                    instrument_name=stem.get("instrumentName", ""),
                    section_name=stem.get("sectionName", ""),
                    section_role=stem.get("sectionRole", ""),
                    bar_count=stem.get("barCount", ""),
                    bpm=stem.get("bpm", ""),
                    key=stem.get("key", ""),
                    uri=stem.get("uri", ""),
                    url=stem.get("url", ""),
                    used_block_ids=stem.get("usedBlockIds", []),
                )
                for stem in section
            ]
            for section in data_schema.chosen_sections
        ],
        generated_stems=[
            Stem(
                id=stem.get("id", ""),
                mix_id=stem.get("mixId", ""),
                dialog_uuid=stem.get("dialogUuid", ""),
                is_original=stem.get("isOriginal", ""),
                is_block=stem.get("isBlock", ""),
                category=stem.get("category", ""),
                caption=stem.get("caption", ""),
                instrument_name=stem.get("instrumentName", ""),
                section_name=stem.get("sectionName", ""),
                section_role=stem.get("sectionRole", ""),
                bar_count=stem.get("barCount", ""),
                bpm=stem.get("bpm", ""),
                key=stem.get("key", ""),
                uri=stem.get("uri", ""),
                url=stem.get("url", ""),
                used_block_ids=stem.get("usedBlockIds", []),
            )
            for stem in data_schema.generated_stems
        ],
        context_song=context_song,
        turn_index=data_schema.turn_index,
        working_section_index=data_schema.working_section_index,
    )


def load_memory_from_cache(memory_id: str) -> MemoryData:
    logger.debug(f"loading memory from cache: {memory_id}")
    memory = cache_get(f"request_answer:{memory_id}")
    if memory:
        logger.debug(f"memory: {memory}")
        memory = json.loads(memory)
    else:
        logger.debug(f"memory not found in cache: {memory_id}")
        raise Exception(f"memory not found in cache: {memory_id}")

    context_song = memory.get("requestInformation", {}).get("contextSongInfo", {})
    context_song = ContextSong(
        song_id=context_song.get("songId", ""),
        bpm=context_song.get("bpm", 0),
        key=context_song.get("key", ""),
        bar_count=context_song.get("barCount", 0),
        section_name=context_song.get("sectionName", ""),
        song_structure=context_song.get("songStructure", ""),
        section_role=context_song.get("sectionRole", ""),
        context_audio_uris=context_song.get("contextAudioUris", []),
        created_sections_order=context_song.get("createdSectionsOrder", []),
        arranged_sections_order=context_song.get("arrangedSectionsOrder", []),
        is_remix=context_song.get("isRemix", False),
    )

    answers = memory.get("answers", [{}])
    if len(answers) == 0:
        sections = [[]]
    else:
        mix = answers[0].get("mix", {})
        if mix:
            sections = mix.get("mixData", {}).get("stems", [])
        else:
            sections = [[]]

    return MemoryData(
        memory_id=memory_id,
        user_prompt=memory.get("chatMessage", {}).get("chatText", ""),
        intent_focused_prompt=memory.get("chatMessage", {}).get(
            "intentFocusedPrompt", ""
        ),
        intent_history=memory.get("context", {}).get("previousContext", []),
        chosen_sections=[
            [
                Stem(
                    id=stem.get("id", ""),
                    mix_id=stem.get("mixId", ""),
                    dialog_uuid=stem.get("dialogUuid", ""),
                    is_original=stem.get("isOriginal", ""),
                    is_block=stem.get("isBlock", ""),
                    category=stem.get("category", ""),
                    caption=stem.get("caption", ""),
                    instrument_name=stem.get("instrumentName", ""),
                    section_name=stem.get("sectionName", ""),
                    section_role=stem.get("sectionRole", ""),
                    bar_count=stem.get("barCount", ""),
                    bpm=stem.get("bpm", ""),
                    key=stem.get("key", ""),
                    uri=stem.get("uri", ""),
                    url=stem.get("url", ""),
                    used_block_ids=stem.get("usedBlockIds", []),
                )
                for stem in section
            ]
            for section in sections
        ],
        generated_stems=[
            Stem(
                id=stem.get("id", ""),
                mix_id=stem.get("mixId", ""),
                dialog_uuid=stem.get("dialogUuid", ""),
                is_original=stem.get("isOriginal", ""),
                is_block=stem.get("isBlock", ""),
                category=stem.get("category", ""),
                caption=stem.get("caption", ""),
                instrument_name=stem.get("instrumentName", ""),
                section_name=stem.get("sectionName", ""),
                section_role=stem.get("sectionRole", ""),
                bar_count=stem.get("barCount", ""),
                bpm=stem.get("bpm", ""),
                key=stem.get("key", ""),
                uri=stem.get("uri", ""),
                url=stem.get("url", ""),
                used_block_ids=stem.get("usedBlockIds", []),
            )
            for stem in memory.get("answers", [{}])[0].get("suggestedStems", [])
        ],
        context_song=context_song,
        turn_index=memory.get("answers", [{}])[0].get("turnIndex", 0),
        working_section_index=memory.get("requestInformation", {}).get(
            "workingSectionIndex", 0
        ),
    )
