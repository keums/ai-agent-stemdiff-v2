import logging

from models import MemoryData, MemorySelection, MemoryStrategy

# from .embedding_logger import log_selected_memory
from .intent_reasoner import make_strategy_with_llm, select_best_memory_with_llm
from .memory_loader import get_recent_valid_dialog_uuids, load_memory_from_cache

CANDIDATE_MEMORY_COUNT = 15
logger = logging.getLogger(__name__)


def select_best_memory(
    user_prompt: str,
    dialog_uuid: str,
    session_uuid: str,
    is_local: bool = True,
) -> tuple[MemoryData, MemoryStrategy, int]:
    """
    Select the best memory data for a music generation session based on user intent.

    This function orchestrates the memory selection process by:
    1. Loading recent memory candidates (either from local files or cache)
    2. Analyzing user intent to determine memory strategy
    3. Selecting appropriate memory data based on the strategy
    4. Returning a MemorySelection object with the chosen memory and strategy

    Args:
        user_prompt (str): The user's current prompt/intent for the music generation
        dialog_uuid (str): Unique identifier for the current dialog session
        session_uuid (str): Unique identifier for the current user session
        is_local (bool, optional): Whether to load memory from local files (True)
                                  or from cache (False). Defaults to True.

    Returns:
        MemorySelection: Object containing the selected memory data, strategy,
                        and metadata for the current session. Returns None if no
                        valid memory is found.

    Note:
        - When is_local=True, memory is loaded from "./output/data_schema" folder
        - When is_local=False, memory is loaded from cache using dialog_uuid and session_uuid
        - The function uses LLM-based reasoning to determine the best memory strategy
        - If strategy indicates starting from scratch, returns minimal memory selection
    """

    recent_memory = None
    recent_id = None
    llm_usage = 0

    candidates_uuids = get_recent_valid_dialog_uuids(
        dialog_uuid, session_uuid, CANDIDATE_MEMORY_COUNT
    )
    logger.debug(f"candidates_uuids: {candidates_uuids}")
    if not candidates_uuids:
        logger.debug("There is no valid memory")
        return (
            MemoryData(),
            MemoryStrategy(
                intent_focused_prompt=user_prompt, is_start_from_scratch=True
            ),
            0,
        )
    else:
        try:
            recent_id = candidates_uuids[0]
            recent_memory = load_memory_from_cache(recent_id)
        except Exception as e:
            logger.info(f"Error loading memory from cache: {e}")
            return (
                MemoryData(),
                MemoryStrategy(
                    intent_focused_prompt=user_prompt, is_start_from_scratch=True
                ),
                0,
            )

    # Decide strategy flags for using last vs. older, and intent (start fresh / suggested stems)
    # 바로 직전 메모리만 사용할 것인지 아니면 이전 메모리도 사용할 것인지 결정
    strategy: MemoryStrategy = make_strategy_with_llm(
        user_prompt,
        previous_intent_focused_prompt=recent_memory.intent_focused_prompt,
        chosen_sections=recent_memory.chosen_sections,
        generated_stems=recent_memory.generated_stems,
        intent_history=recent_memory.intent_history,
        working_section_index=recent_memory.working_section_index,
    )
    llm_usage += 1

    # Pick strategy
    if strategy.should_load_older_memories:
        strategy.selected_older_memory_ids = [
            select_best_memory_with_llm(
                user_prompt,
                strategy.intent_focused_prompt,
                [load_memory_from_cache(uuid) for uuid in candidates_uuids],
            )["selected_previous_memory"]
        ]
        llm_usage += 1

    return recent_memory, strategy, llm_usage
