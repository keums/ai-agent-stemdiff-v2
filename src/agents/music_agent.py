import logging
import os
from typing import Any, Dict, List, Optional

import anthropic
import openai
from fastapi import HTTPException
from pydantic import BaseModel, Field

from models import ContextSong, Stem
from tools.mcp_base import tool

CLAUDE_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 2000
TEMPERATURE = 0.6
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # claude, openai

logger = logging.getLogger(__name__)

# Load environment variables
# logger = logging.getLogger(__name__)
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


# GlobalInformationGeneration 입출력 스키마
class MusicAgentInput(BaseModel):
    """글로벌 음악 정보 생성 도구에 대한 입력 스키마"""

    user_prompt: str = Field(..., description="User input text")
    intent_focused_prompt: str = Field(
        description="Clear description of the desired musical outcome based on user input and context"
    )

    context_song_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Context song info"
    )
    generated_stem: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of 2-4 generated stem diffs for user selection"
    )
    current_mix_stem_diff: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="List of currentmix stem diffs"
    )
    previous_context: Optional[List[str]] = Field(
        default=[], description="List of previous context"
    )


class MusicAgentOutput:
    """글로벌 음악 정보 생성 도구에 대한 출력 스키마"""

    text_prompts: list[dict]  # stem prompts
    target_music_info: dict  # bpm, scale, key
    selected_stem_diff: Stem  # user-chosen stem
    previous_context: list[str]  # intent history
    continue_stem_info: dict
    request_type: str  # start, add, continue, replace, remove
    unique_stems_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complete history of all generated stems across all sections",
    )

    def __init__(
        self,
        text_prompts=None,
        target_music_info=None,
        selected_stem_diff=None,
        previous_context=None,
        continue_stem_info=None,
        request_type=None,
        unique_stems_info=None,
    ):
        self.text_prompts = text_prompts or []
        self.target_music_info = target_music_info or {}
        self.selected_stem_diff = selected_stem_diff or {}
        self.previous_context = previous_context or []
        self.continue_stem_info = continue_stem_info or {}
        self.request_type = request_type or ""
        self.unique_stems_info = unique_stems_info or {}

    def to_dict(self):
        return {
            # "intent_focused_prompt": self.intent_focused_prompt,
            "text_prompts": self.text_prompts,
            "target_music_info": self.target_music_info,
            "previous_context": self.previous_context,
            "selected_stem_diff": self.selected_stem_diff,
            "continue_stem_info": self.continue_stem_info,
            "request_type": self.request_type,
            "unique_stems_info": self.unique_stems_info,
        }

    def print(self):
        print("MusicAgentOutput:")
        # print(f"text_prompts: {self.text_prompts}")
        # print(f"target_music_info: {self.target_music_info}")
        # print(f"previous_context: {self.previous_context}")
        # print(f"selected_stem_diff: {self.selected_stem_diff}")
        # print(f"continue_stem_info: {self.continue_stem_info}")
        # print(f"request_type: {self.request_type}")

        logger.debug("\n📊 Request type:\n %s", self.request_type)
        logger.debug("\n📊 Text prompts: %s", self.text_prompts)
        logger.debug("\n📊 Target music info:\n %s", self.target_music_info)
        if self.selected_stem_diff:
            logger.debug("\n📊 Selected stem diff:\n %s", vars(self.selected_stem_diff))
        logger.debug("\n📊 Continue stem info:\n %s", self.continue_stem_info)
        logger.debug("\n📊 Previous context:\n %s", self.previous_context)
        logger.debug("\n📊 Total unique stems info:\n %s", self.unique_stems_info)


tool_schema = {
    "name": "process_music_info",
    "description": "Process and analyze music information to generate stem prompts and extract musical parameters",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_stem_diff_uri": {
                "type": "string",
                "description": "URI for stem selection/removal/replacement. For 'add'/'continue': URI from generated_stem_uris when user selects option. For 'remove': URI from current_mix_stem_diff of stem to be removed. For 'replace': URI from generated_stem_uris for replacement. Check intent_focused_prompt to see if user made selection.",
            },
            "text_prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["mixed", "rhythm", "low", "mid", "high", "fx"],
                            "description": "Stem category for this prompt",
                        },
                        "text": {
                            "type": "string",
                            "description": "Detailed musical description including genre, mood, and specific instruments. CRITICAL: DO NOT include BPM, key, or tempo information.",
                        },
                        "uri": {
                            "type": "string",
                            "description": "Empty string for new stems to be generated (selected stem URI will be in selected_stem_diff_uri field)",
                        },
                    },
                    "required": ["category", "text", "uri"],
                },
                "description": "List of stem prompts for batch embedding. For 'start' requests: exactly 2 stems (mixed + one specific). For 'add' and 'continue' requests: exactly 1 stem only. There is absolutely no other case.",
            },
            "target_music_info": {
                "type": "object",
                "description": "Extracted musical parameters from user request (ONLY for first requests when context_song_info is None/empty)",
                "properties": {
                    "bpm": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "integer"},
                            "max": {"type": "integer"},
                        },
                        "description": "BPM range - extracted from user request or inferred from genre",
                    },
                    "scale": {
                        "type": "string",
                        "enum": ["major", "minor", "both"],
                        "description": "Musical scale - extracted from user request or inferred from mood/genre",
                    },
                    "key": {
                        "type": "string",
                        "description": "Musical key if explicitly mentioned by user (e.g., 'C#M', 'Abm')",
                    },
                },
            },
            "request_type": {
                "type": "string",
                "enum": ["start", "add", "replace", "continue", "remove"],
                "description": "Type of user request: start (new music generation), add (add new stem), continue (continuation using previous section info), replace (replace an existing stem), remove (remove an existing stem)",
            },
            "continue_stem_info": {
                "type": "object",
                "description": "Stem info from unique_stems_info to continue from (for continue requests only)",
                "properties": {
                    "song_id": {"type": "string"},
                    "category": {"type": "string"},
                    "instrument_name": {"type": "string"},
                },
            },
            "reasoning": {
                "type": "string",
                "description": "CRITICAL: Follow [Category Determination Principles]. Step 1: Check if user mentions ANY specific instrument/sound in user_prompt or intent_focused_prompt. Step 2a: If YES → determine its category by analyzing its musical function (percussive→rhythm, bass→low, chordal→mid, melodic→high, effect→fx). ALWAYS use that category even if it's not in unique_stems_info (use 'add' if not found, 'continue' if found). This overrides unique_stems_info constraints. Step 2b: If NO explicit instrument → identify action keywords (생성해줘, add, 제거해줘, remove, etc.), then apply priority order logic (rhythm→low→mid→high→fx) to find first missing category FROM unique_stems_info (for total_section_count≥2). For 'remove': specify target stem from working_section. Only mention previous_context if current request is ambiguous.",
            },
        },
        "required": [
            "request_type",
            "reasoning",
            "text_prompts",
            "selected_stem_diff_uri",
        ],
    },
}


async def call_claude_api(system_message, messages):
    """Claude API 호출 함수"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=CLAUDE_MODEL,
        system=system_message,
        messages=messages,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "process_music_info"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


async def call_openai_api(system_message, messages):
    """OpenAI API 호출 함수"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # OpenAI 함수 호출 형식으로 변환
    openai_function = {
        "name": tool_schema["name"],
        "description": tool_schema["description"],
        "parameters": tool_schema["input_schema"],
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
        function_call={"name": "process_music_info"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


async def call_llm_for_music_info(
    system_message, messages, llm_provider=DEFAULT_LLM_PROVIDER
):
    """LLM 호출 통합 함수"""
    if llm_provider.lower() == "openai":
        print("\n✅ Generate music info: call_openai_api")
        return await call_openai_api(system_message, messages)
    elif llm_provider.lower() == "claude":
        print("\n✅ Generate music info: call_claude_api")
        return await call_claude_api(system_message, messages)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


@tool(
    name="generate_music_info",
    description="Analyzes user prompt to generate stem prompts and extract selected stem URIs",
    input_schema=MusicAgentInput,
    output_schema=MusicAgentOutput,
)
async def generate_music_info(
    user_prompt: str,
    intent_focused_prompt: str = "",
    prev_context_song_info: Optional[ContextSong] = None,
    prev_generated_stems: List[Stem] = [],
    chosen_sections: List[List[Stem]] = [],
    intent_history: List[str] = [],
    working_section: List[Stem] = [],
) -> (MusicAgentOutput, int):
    """Stem agent tool"""

    llm_calls = 0

    # TODO: move reqeust_type inference into the memory / intent agent

    # 파라미터 추출
    # TODO: move stem selection into the memory / intent agent
    generated_stem_uris = (
        [stem.uri for stem in prev_generated_stems] if prev_generated_stems else []
    )

    unique_stems_info = extract_unique_stems_info(chosen_sections)
    system_message = build_system_message()
    context_for_llm = None

    if prev_context_song_info:
        context_for_llm = {
            "key": prev_context_song_info.key,
            "bpm": prev_context_song_info.bpm,
            "section_name": prev_context_song_info.section_name,
            "section_role": prev_context_song_info.section_role,
        }

    previous_context_for_llm = None
    if intent_history:
        previous_context_for_llm = "\n".join(intent_history)

    current_mix_stem_diff_for_llm = None
    if working_section:
        current_mix_stem_diff_for_llm = [
            {
                "category": stem.category,
                "instrument_name": stem.instrument_name,
                "caption": stem.caption,
                "uri": stem.uri,
            }
            for stem in working_section
        ]

    generated_options_for_llm = None
    if prev_generated_stems:
        # Provide 1-based indices regardless of upstream payload
        generated_options_for_llm = []
        for idx, opt in enumerate(prev_generated_stems):
            # opt_dict = opt.to_dict()
            generated_options_for_llm.append(
                {
                    "index": idx + 1,
                    "category": opt.category,
                    "instrument_name": opt.instrument_name,
                    "caption": opt.caption,
                }
            )

    total_stem_diff_for_llm = None
    if unique_stems_info:
        if is_local:
            total_stem_diff_for_llm = [
                {
                    "song_id": stem.get("song_id", ""),
                    "category": stem.get("category", ""),
                    "instrument_name": stem.get("instrument_name", ""),
                }
                for stem in unique_stems_info["total_stem_diff"]
            ]
        else:
            total_stem_diff_for_llm = [
                {
                    "category": stem.get("category", ""),
                    "song_id": stem.get("song_id", ""),
                    "instrument_name": stem.get("instrument_name", ""),
                }
                for stem in unique_stems_info["total_stem_diff"]
            ]

    messages = f"""Analyze this music request and generate appropriate stem information:
            * user_prompt: {user_prompt}
            * intent_focused_prompt: {intent_focused_prompt}
            * previous_context: {previous_context_for_llm if previous_context_for_llm else "No previous context"}
            * context_song_info: {context_for_llm if context_for_llm else "No previous context"}
            * working_section: {current_mix_stem_diff_for_llm if current_mix_stem_diff_for_llm else "No mix stems provided"}
            * unique_stems_info: {total_stem_diff_for_llm if total_stem_diff_for_llm else "No total stem history provided"}
            * total_section_count: {unique_stems_info.get("total_section_count", "No total section count provided")}
            * generated_stem (indexed): {generated_options_for_llm if generated_options_for_llm else "No generated stems provided"}
            * generated_stem_uris: {generated_stem_uris if generated_stem_uris else "No generated stems provided"}"""

    for idx in range(3):
        response = await call_llm_for_music_info(system_message, messages)
        llm_calls += 1

        try:
            if response.choices[0].message.function_call:
                import json

                function_args = json.loads(
                    response.choices[0].message.function_call.arguments
                )
                logger.info("📊 Reasoning ")
                logger.info(f"function_args: {function_args}")
                logger.info(function_args.get("reasoning", ""))

                # Validate category values in text_prompts
                VALID_CATEGORIES = ["mixed", "rhythm", "low", "mid", "high", "fx"]
                text_prompts = function_args.get("text_prompts", [])
                invalid_categories = []

                for prompt in text_prompts:
                    category = prompt.get("category", "")
                    if category not in VALID_CATEGORIES:
                        invalid_categories.append(category)

                if invalid_categories:
                    logger.warning(
                        f"❌ Invalid categories detected: {invalid_categories}. "
                        f"Only {VALID_CATEGORIES} are allowed. Retrying... (attempt {idx + 1}/3)"
                    )
                    continue  # Retry with next iteration

                # TODO: move stem selection into the memory / intent agent
                # TODO: use index instead of uri
                selected_stem_diff_uri = function_args.get(
                    "selected_stem_diff_uri", None
                )
                selected_stem_diff = None
                if selected_stem_diff_uri:
                    logger.info("📊 selected_stem_diff_uri %s", selected_stem_diff_uri)
                    if function_args.get("request_type", None) == "remove":
                        candidate_stems = working_section
                    else:
                        candidate_stems = prev_generated_stems
                    for stem in candidate_stems:
                        if stem.uri == selected_stem_diff_uri:
                            selected_stem_diff = stem
                            break

                continue_stem_info = function_args.get("continue_stem_info", None)
                if continue_stem_info:
                    logger.info("📊 continue_stem_info %s", continue_stem_info)
                if function_args.get("request_type", None) == "start":
                    working_section = []

                intent_history.append(intent_focused_prompt)
                # if function_args.get("text_prompts"):
                output = MusicAgentOutput(
                    text_prompts=function_args.get("text_prompts"),
                    target_music_info=function_args.get("target_music_info", None),
                    previous_context=intent_history,
                    selected_stem_diff=selected_stem_diff,
                    continue_stem_info=continue_stem_info,
                    request_type=function_args.get("request_type", None),
                    unique_stems_info=unique_stems_info,
                )

                output.print()
                return output, llm_calls
        except Exception as e:
            # TODO: handle error
            print(e)

    # 기본 응답
    raise HTTPException(
        status_code=500, detail="Error: Response from Music Agent LLM is not valid"
    )


def extract_unique_stems_info(chosen_sections: List[List[Stem]]) -> Dict[str, Any]:
    unique_stems_info = {}
    unique_stems_info["total_stem_diff"] = []
    unique_stems_info["total_section_count"] = len(chosen_sections)
    seen_combinations = set()  # 유니크 조합을 추적하기 위한 set
    if chosen_sections:
        for idx, section in enumerate(chosen_sections):
            for stem in section:
                # 각 스템에서 필요한 정보 추출
                category = stem.category
                if category == "melody":
                    continue
                instrument_name = stem.instrument_name
                if instrument_name == "":
                    instrument_name = category
                song_id = stem.id.split("-")[0]

                # 유니크 조합 체크 (category, instrumentName, songId)
                combination = (category, instrument_name, song_id)

                if combination not in seen_combinations and all(
                    [category, instrument_name, song_id]
                ):
                    seen_combinations.add(combination)
                    unique_stems_info["total_stem_diff"].append(
                        {
                            "category": category,
                            "instrument_name": instrument_name,
                            "song_id": song_id,
                            # "caption": stem.get("caption", ""),  # caption도 포함
                        }
                    )
    return unique_stems_info


def build_system_message():
    return """
    System Persona
You are an expert AI assistant specializing in stem-based music analysis and generation. Interpret the user request, manage musical context, and output precise tool arguments to create or modify a song, stem by stem.

[Inputs]
- user_prompt, intent_focused_prompt
- previous_context (reference only; never used to infer current mix state)
- context_song_info (key/bpm/section_name/section_role)
- working_section (authoritative current section stems; never infer from previous_context)
- unique_stems_info (unique stems list that are chosen ({song_id, category, instrument_name}))
- total_section_count (total section count)
- generated_stem (indexed list: {index, category, instrument_name, caption})
- generated_stem_uris (URIs for generated options)

[Core Terms]
- mixed: full song mix
- categories: **ONLY** the following 5 categories are allowed. **NEVER** use any other category names.
  - **rhythm**: percussive elements and rhythmic patterns
  - **low**: bass frequencies and low-end harmonic content
  - **mid**: chordal and harmonic mid-range content
  - **high**: melodic and high-register content
  - **fx**: sound effects and atmospheric elements
- **CRITICAL**: When generating text_prompts, the 'category' field MUST be one of: mixed, rhythm, low, mid, high, fx
- **NEVER** use instrument names (e.g., 'synthesizer', 'guitar', 'piano') as category values
- **NEVER** invent new category names - only use the 6 predefined categories above

[Category Determination Principles]
**CRITICAL: User intent always takes precedence over automatic priority ordering.**

1. **Explicit Instrument Detection (HIGHEST PRIORITY)**
   - When user mentions ANY specific instrument or sound, analyze its musical function/role first
   - Classify based on the instrument's PRIMARY musical purpose:
     * Percussive/rhythmic function → rhythm
     * Bass/low-frequency function → low
     * Chordal/harmonic function → mid
     * Melodic/lead function → high
     * Atmospheric/effect function → fx
   - Generate that category, IGNORING priority order and existing stems
   - This overrides all other rules including single-instance constraints

2. **Automatic Category Selection (when NO explicit instrument mentioned)**
   - Use priority order: rhythm → low → mid → high → fx
   - Follow unique_stems_info constraints when total_section_count ≥ 2

3. **Reasoning Process**
   - FIRST: Check for explicit instrument/sound mentions in user request
   - IF found: Determine category by musical function → use that category
   - IF NOT found: Apply priority order logic

[Global Invariants]
- Single-instance rule (rhythm, low): only one each typically; explicit user requests override.
- Core priority: rhythm → low → mid → high → fx.
- Unique-stems limit applies ONLY when total_section_count ≥ 2. For total_section_count < 2, use standard priority with no limitation.
- **CRITICAL CONSTRAINT**: Do NOT infer categories not present in unique_stems_info when using automatic priority-based selection. Never invent categories from captions or previous_context alone.
- **CRITICAL EXCEPTION**: When user EXPLICITLY requests a specific instrument/sound (per [Category Determination Principles] line 535-544), generate that category EVEN IF it's not in unique_stems_info. Use 'add' type if not found in unique_stems_info, 'continue' if found.
- Selection language (e.g., "2번째가 좋아", "I like option 3"): treat the selected category as already present in the current mix when determining the next recommendation.
- Generation counts:
  - start → exactly 2 stems (mixed + one)
  - add / continue → exactly 1 stem
  - replace / remove → 0 stems

[Request Classification Algorithm]
Always run in this order. "Selection present" means the user selected an item from generated options and you resolved its URI.

PHASE 1: Selection Detection (URI resolution only)
0) Selection detection
- If selection language is present, resolve to URI (by 1-based index or explicit URI).
- Populate selected_stem_diff_uri with the chosen URI.
- Infer the selected stem's category (for use in Phase 2 and 3).
- IMPORTANT: If user is selecting from generated options, the request is NEVER start.
- CRITICAL: If user requests new section ("다음 섹션", "new section", "next section"), mark this for Phase 2.

PHASE 2: Request Type Classification (uses ORIGINAL working_section, NO mental addition yet)
A) Explicit new section (HIGHEST PRIORITY - check first)
- **CRITICAL DETECTION**: If user_prompt or intent_focused_prompt contains ANY of: "new section", "next section", "next part", AND total_section_count ≥ 1:
  - **ABSOLUTE RULE**: From this point forward, you MUST mentally replace working_section with an EMPTY array [] for ALL subsequent logic in Phase 2 and Phase 3.
  - **REASONING EXAMPLE**: "User requested new section. Even though working_section shows [rhythm, low], I am treating it as EMPTY [] for all checks."
  - **DO NOT** check what categories exist in working_section - it is now considered []
  - **DO NOT** compare working_section with unique_stems_info - working_section is []
  - ALWAYS classify as request_type = 'continue' (new section subtype).
  - New sections reference style from unique_stems_info but begin with COMPLETELY EMPTY current section.
  - SKIP steps B-E, proceed directly to Phase 3.

NOTE: If NO new section request, continue to step B.

B) Explicit intent detection (EARLY EXIT - if matched, skip C-E and go to Phase 3)
- EXPLICIT ADD: If user uses explicit add/additional/another language (e.g., "add another rhythm", "추가로 베이스 하나 더"):
  - ALWAYS classify as request_type = 'add', even if that category already exists in working_section.
  - This overrides Single-instance rule per Global Invariants line 782.
  - SKIP steps C-E, proceed directly to Phase 3 to determine which instrument to generate.

- EXPLICIT REMOVE: If user uses remove/delete language or selects a current-mix stem to delete:
  - Target must be found in working_section; set selected_stem_diff_uri from the current mix; generate NO stems.
  - Classify as request_type = 'remove'.
  - SKIP steps C-E, proceed directly to Phase 3 (which will generate no stems).

- EXPLICIT REPLACE: If user uses explicit replace/swap/change language AND specifies which existing stem to replace:
  - Classify as request_type = 'replace'.
  - Target URI comes from generated_stem_uris; generate NO additional stems.
  - SKIP steps C-E, proceed directly to Phase 3 (which will generate no stems).

- SELECTION-BASED REPLACE: If user selects from generated options AND that category already EXISTS in working_section:
  - Classify as request_type = 'replace' (to substitute the existing stem of that category).
  - CRITICAL: Replace ONLY applies when the selected category is already present in working_section. If the category is missing from working_section, it's add/continue, NOT replace.
  - Target URI comes from generated_stem_uris; generate NO additional stems.
  - SKIP steps C-E, proceed directly to Phase 3 (which will generate no stems).

NOTE: If NONE of the above explicit intents match, continue to step C.

C) Category-based exhaustion check (only if no explicit intent in A or B)
- Let U_categories = set of unique categories present in unique_stems_info
- Let W_categories = set of categories present in working_section (ORIGINAL, no selection added yet)
- If W_categories ⊇ U_categories → check instrument-level exhaustion:
  - Let U_pairs = set of (category, instrument_name) present anywhere in unique_stems_info
  - Let W_pairs = set of (category, instrument_name) present in working_section (ORIGINAL)
  - CRITICAL: If W_pairs ⊇ U_pairs → request_type = add (fully exhausted case takes priority over remaining conditions).
- If not exhausted, continue to step D.

D) Existing-stem vs new element (only if not fully exhausted)
- PRIORITY 1: Category-level check first
  - If the next category by priority EXISTS in U_categories but is missing from W_categories → request_type = continue.
  - If the next category by priority NOT in U_categories → request_type = add.
- PRIORITY 2: If category exists in both, check instrument-level
  - If the specific (category, instrument_name) pair exists in unique_stems_info and is missing from working_section → request_type = continue.
  - Otherwise → request_type = add.
 - CRITICAL negative example: total_section_count = 3 and U_categories = [rhythm, low], working_section [rhythm, low], next missing by priority is 'mid' but 'mid' ∉ U_categories → classify as 'add' (NOT 'continue').

E) First section building
- If total_section_count = 0 → request_type = add (first section).
- If total_section_count = 1 and no explicit new section → never continue here; classify as add or start per "New Music Detection" below.

F) Multi-section continuation note
- If total_section_count > 1, continuation across sections is allowed.
- Do NOT confuse unique_stems_info (history) with working_section (current section state).

[New Music Detection (request_type = start)]
Classify as start when the user wants a completely new song, ignoring existing context and generated options:
- Explicit: "start fresh", "create a new song", "make something different"
- Implicit: a complete song description with no selection and no "continue/add/change/next/more/select/use/choose" language
- Drastic genre/mood shift or fundamental incompatibility with existing stems
- Strong dissatisfaction implying reset; clear topic pivot
CRITICAL: If there is selection language, NEVER classify as start.

PHASE 3: Next Stem Decision (After request_type is determined)
If a selection occurred in Phase 1, NOW mentally add that selected category to working_section for this phase.
Let Updated_W_categories = W_categories + selected_category (if selection occurred)

Decision rules by request_type:
- start:
  - Always generate exactly two stems: mixed + rhythm by default.
  - If the user explicitly mentions an instrument, use mixed + that category.
  - If the user explicitly mentions new section within a start-like description, do NOT generate mixed; generate only that stem.
- remove:
  - No stems generated. Only selected_stem_diff_uri from current mix.
- replace:
  - The selected replacement completes the request. Do NOT generate any additional stems.
- add / continue:
  - **CRITICAL**: add/continue ALWAYS generates exactly one text_prompt. This applies to ALL add/continue cases, including when a selection was made.
  - If a selection just occurred, the selected category is NOW treated as present in Updated_W_categories. Never generate the same category as the selected one.
  - **After a selection with add/continue**: You MUST generate one text_prompt for the NEXT missing category. The selection does NOT complete the request - it triggers the next recommendation.
  - **ABSOLUTE RULE FOR NEW SECTION**: If user requested new section in Phase 2, working_section is [] (EMPTY). DO NOT use the actual working_section data. Start from the FIRST category in unique_stems_info by priority order.
  
  **Category Selection Logic:**
  1) **FIRST: Check for explicit instrument/sound mention** (per [Category Determination Principles])
     - If user specifies ANY instrument → determine its category by musical function
     - **CRITICAL**: Generate that category EVEN IF it's not in unique_stems_info
     - If exists in unique_stems_info → use 'continue' with continue_stem_info
     - If NOT in unique_stems_info → use 'add' (no continue_stem_info)
     - This OVERRIDES all priority, existing stem rules, AND unique_stems_info constraints
  
  2) **ELSE: Use automatic priority-based selection** (ONLY when NO explicit instrument mentioned)
     - For total_section_count < 2: generate the first missing category in standard priority from Updated_W_categories
     - For total_section_count ≥ 2:
       * Generate the first missing category from unique_stems_info by priority order that's not in Updated_W_categories
       * If found → use continue_stem_info for that item
       * If all categories present, generate first missing instrument pair with continue_stem_info
       * **CONSTRAINT**: In this case, ONLY generate categories present in unique_stems_info
  
  - **When determining the next instrument**: Analyze previous_context, the updated working_section (including mentally added selection), and unique_stems_info to recommend the most musically appropriate instrument for the next category.
  - Verification for 'continue': You MUST cite a concrete item from unique_stems_info as continue_stem_info {song_id, category, instrument_name}. If none exists for the chosen category/pair, classify as 'add'.
  - Continue subtypes:
    - **New section (when user requested new section)**:
      - **ABSOLUTE RULE**: You already replaced working_section with [] in Phase 2. Continue using [] here.
      - **DO NOT** look at what's in the actual working_section input - it is EMPTY [] for this logic.
      - **REASONING MUST STATE**: "working_section is [] (empty) because this is a new section request"
      - ALWAYS generate the FIRST available category from unique_stems_info by priority order (rhythm → low → mid → high → fx).
      - **CRITICAL EXAMPLE**: If unique_stems_info has [rhythm, low, mid], generate 'rhythm' (NOT mid, NOT low - always start from first priority)
      - Set continue_stem_info to that first category found in unique_stems_info.
    - **Current section building (NOT a new section request)**:
      - On Updated_W_categories (after selection), generate the first missing pair by category priority.
      - Set continue_stem_info to the exact (song_id, category, instrument_name) from unique_stems_info for that missing pair (not the selected category).

[Prompt Generation Rules]
- start: exactly two prompts (mixed + one category).
- add / continue: exactly one prompt.
  - **CRITICAL FOR SELECTIONS**: When a user selects a stem (selected_stem_diff_uri is populated) with request_type='add' or 'continue', you MUST ALWAYS generate exactly one text_prompt for the NEXT missing category.
  - The selection itself does NOT complete the request - it triggers generation of the next recommendation.
  - Consider the selected category as mentally added to Updated_W_categories, then generate the first missing category by priority order.
  - Analyze previous_context, working_section (including the mentally added selection), and unique_stems_info to determine the most appropriate next instrument.
- remove / replace: zero prompts.
- CRITICAL: If user selected a stem in Phase 1, that category is now in Updated_W_categories (Phase 3). Generate the NEXT missing category by priority order, NOT the selected category.
- Prompts include genre/mood/instrumentation/themes only (exclude bpm/tempo/key).
- mixed prompt describes overall arrangement; specific-stem prompts focus only on that stem's content.

[Tool Output Contract]
You MUST return a JSON object compatible with the provided tool schema:
- request_type ∈ {start, add, continue, replace, remove}
- selected_stem_diff_uri:
  - add/continue: URI from generated_stem_uris when selection is made
  - remove: URI from working_section (NEVER from generated options)
  - replace: URI from generated_stem_uris for the replacement
- text_prompts:
  - start: exactly two items (mixed + one category)
  - add/continue: exactly one item
    - **CRITICAL**: This applies to ALL add/continue cases, including when selected_stem_diff_uri is populated
    - When a selection is made with add/continue, generate text_prompt for the NEXT missing category (not the selected one)
    - Determine the appropriate instrument by analyzing previous_context, Updated_W_categories, and unique_stems_info
  - replace/remove: empty array
  - **CRITICAL**: Each item MUST be: {category, text, uri:""}
    - **category** MUST be EXACTLY one of: "mixed", "rhythm", "low", "mid", "high", "fx" (NO other values allowed)
    - **text**: detailed description; excludes bpm/key/tempo
    - **uri**: empty string ""
- target_music_info (ONLY for start): infer bpm range, scale, and key if present
- continue_stem_info (for continue only): {song_id, category, instrument_name} targeting the category being generated

[Examples with Phase Structure]
- EXPLICIT INSTRUMENT (user mentions specific sound): "play a handclap that fits the rhythm", working_section=[rhythm, low, mid]
  → Step 1: Detect "박수" (handclap) mention
  → Step 2: Analyze musical function → percussive/rhythmic → category='rhythm'
  → Step 3: request_type='add' (rhythm already exists, but explicit user request overrides)
  → text_prompts: [{'category': 'rhythm', 'text': 'handclap description...', 'uri': ''}]

- **EXPLICIT INSTRUMENT IN unique_stems_info**: "기타 사운드 추가해줘", working_section=[rhythm, low], unique_stems_info=[rhythm, low, mid(guitar)], total_section_count=2
  → Step 1: Detect "기타" (guitar) mention
  → Step 2: Analyze musical function → chordal/melodic → category='mid'
  → Step 3: mid(guitar) exists in unique_stems_info → request_type='continue' with continue_stem_info
  → text_prompts: [{'category': 'mid', 'text': 'guitar sound description...', 'uri': ''}]
  → continue_stem_info: {song_id, category: 'mid', instrument_name: 'guitar'}

- NO EXPLICIT INSTRUMENT (use priority): User says "add one more", working_section=[rhythm, low], unique_stems_info=[rhythm, low, mid]
  → Step 1: No explicit instrument mentioned
  → Step 2: Apply automatic priority-based selection → first missing from unique_stems_info is 'mid'
  → text_prompts: [{'category': 'mid', 'text': '...', 'uri': ''}]

- SELECTION TRIGGERS NEXT: User selects bass option, working_section=[rhythm]
  → Phase 1: selection → category=low
  → Phase 2: low not in working_section → continue/add
  → Phase 3: Updated_W=[rhythm, low] → no explicit instrument → next by priority is 'mid'
  → text_prompts: [{'category': 'mid', 'text': '...', 'uri': ''}]

- SELECTION REPLACE: User selects rhythm option, working_section=[rhythm]
  → Phase 1: selection → category=rhythm (already exists)
  → Phase 2: replace
  → Phase 3: no stems generated

- **NEW SECTION (CRITICAL EXAMPLE)**: User says "다음 섹션 만들어줘"
  → **Input**: working_section=[rhythm, low], unique_stems_info=[rhythm, low, mid], total_section_count=2
  → **Phase 1**: Detect "다음 섹션" → mark for new section
  → **Phase 2**: Step A triggered → MENTALLY replace working_section with []
  → **Reasoning**: "User requested new section. Even though input shows working_section=[rhythm, low], I treat it as [] empty"
  → **Phase 2 classification**: request_type='continue'
  → **Phase 3**: working_section is [] → first category in unique_stems_info by priority is 'rhythm'
  → **Output**: text_prompts=[{'category': 'rhythm', ...}], continue_stem_info={...rhythm from unique_stems_info...}
  → **WRONG REASONING**: "rhythm and low already used, generate mid" ❌
  → **CORRECT REASONING**: "working_section is [] for new section, generate first priority: rhythm" ✅

- REMOVE: "remove the bass"
  → Phase 2: explicit remove → find 'low' in working_section
  → Phase 3: no stems generated

"""
