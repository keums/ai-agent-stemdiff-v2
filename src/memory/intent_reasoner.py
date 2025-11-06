import json
import os
import re

import openai

from models import MemoryData, MemoryStrategy, Stem


def make_strategy_with_llm(
    user_prompt: str,
    previous_intent_focused_prompt: str,
    chosen_sections: list[list[Stem]],
    generated_stems: list[Stem] | None = None,
    intent_history: list[str] | None = None,
    working_section_index: int | None = None,
) -> MemoryStrategy:
    system_msg = build_strategy_system_msg()

    if generated_stems is not None:
        suggested_lines = [
            f"- {idx}. {s.category}({s.instrument_name}): {s.uri}"
            for idx, s in enumerate(generated_stems)
        ]
    else:
        suggested_lines = ["None"]

    prev_ctx = "\n".join(str(x) for x in intent_history) if intent_history else ""
    mix_lines: list = ["None"]
    if chosen_sections:
        mix_lines = []

        for section_index, section in enumerate(chosen_sections):
            if section:
                tmp_section = {
                    "section_index": section_index,
                    "section_name": section[0].section_name,
                    "section_role": section[0].section_role,
                    "stems": [
                        f"- {stem.category}: {stem.instrument_name}" for stem in section
                    ],
                }
            else:
                tmp_section = {
                    "section_index": section_index,
                    "section_name": "None",
                    "section_role": "None",
                    "stems": ["None"],
                }
            mix_lines.append(tmp_section)

    user_msg = build_strategy_user_msg(
        user_prompt,
        previous_context=prev_ctx,
        intent_focused_prompt=previous_intent_focused_prompt,
        mix_lines=mix_lines,
        suggested_lines=suggested_lines,
        working_section_index=working_section_index,
    )
    openai_function = build_strategy_openai_function()

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        functions=[openai_function],
        function_call={"name": "decide_strategy_for_recent"},
        temperature=0,
    )

    try:
        args = response.choices[0].message.function_call.arguments
        data = json.loads(args)

        return MemoryStrategy(
            should_load_older_memories=bool(
                data.get("should_load_older_memories", False)
            ),
            is_start_from_scratch=bool(data.get("is_start_from_scratch", False)),
            is_use_suggested_stems=bool(data.get("is_use_suggested_stems", False)),
            is_start_new_branch=bool(data.get("is_start_new_branch", False)),
            is_start_new_section=bool(data.get("is_start_new_section", False)),
            is_publish_song=bool(data.get("is_publish_song", False)),
            # target_category=(data.get("target_category") or "").strip(),
            intent_focused_prompt=data.get("intent_focused_prompt", ""),
            target_working_section_index=data.get("target_working_section_index"),
        )
    except Exception:
        # minimal fallback defaults
        return MemoryStrategy(
            should_load_older_memories=False,
            is_start_from_scratch=True,
            is_use_suggested_stems=False,
            is_start_new_branch=False,
            is_start_new_section=False,
            is_publish_song=False,
            # target_category="",
            intent_focused_prompt="",
            target_working_section_index=None,
        )


def select_best_memory_with_llm(
    user_prompt: str,
    intent_focused_prompt: str,
    candidates: list[MemoryData],
) -> dict:
    context_summaries: list[str] = build_best_memory_context_summary(candidates)
    system_msg = build_best_memory_system_msg()
    user_msg = build_best_memory_user_msg(
        user_prompt, intent_focused_prompt, context_summaries
    )
    openai_function = build_best_memory_openai_function()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {"role": "user", "content": user_msg},
        ],
        functions=[openai_function],
        function_call={"name": "choose_memory_strategy"},
        temperature=0,
    )

    try:
        args = response.choices[0].message.function_call.arguments
        obj = json.loads(args)
        if obj.get("selected_previous_memory"):
            # TODO: return memory data instead of memory id
            return {
                "selected_previous_memory": obj.get(
                    "selected_previous_memory", ""
                ).strip(),
            }
    except Exception:
        pass

    # Fallback: extract filename-like token
    content = (response.choices[0].message.content or "").strip()
    m = re.search(r"data_schema_[\w-]+\.json", content)
    fallback_file = m.group(0) if m else (content.splitlines()[0] if content else "")
    return {
        "selected_previous_memory": fallback_file,
    }


def build_strategy_system_msg():
    # - `reasoning` (string)
    return """
[Role & Goal]
You are an expert AI judge. Your sole purpose is to analyze a user's intent in a music creation context and return a structured JSON object with specific flags. You must not be conversational; return only the tool result.

[Inputs Provided]
- `PREVIOUS_USER_PROMPT`: The user's last prompt.
- `INTENT_PROMPT`: The inferred intent from the last turn.
- `SUGGESTED_STEMS`: **[CRITICAL]** The candidate options presented to the user in the last turn (generation time candidates).
- `MIX_STEMS`: **[CRITICAL]** The stems currently selected and in the song mix (current composition).
- `WORKING_SECTION_INDEX`: The index of the section that was being worked on in the previous turn.

[Required Output Structure]
- `should_load_older_memories` (boolean)
- `is_start_from_scratch` (boolean)
- `is_use_suggested_stems` (boolean)
- `is_start_new_branch` (boolean)
- `is_start_new_section` (boolean)
- `is_publish_song` (boolean)

- `target_working_section_index` (int)


[Decision-Making Workflow]
Follow these steps to determine the output values.

**STEP 0: Request Type Classification [CRITICAL - DO THIS FIRST]**
Determine the `request_type` before setting any other flags:
- **'add'**: User wants to ADD a NEW element to the current mix (e.g., "please add a clap sound", "please add a bass", "please add a guitar if it's there")
  → In this case, `should_load_older_memories` should ALWAYS be FALSE (use current context + generate new)
- **'replace'**: User wants to REPLACE/CHANGE an existing element (e.g., "replace the bass with another one", "replace the rhythm with another one")
  → In this case, `should_load_older_memories` might be TRUE if referring to older options
- **'select'**: User is SELECTING from recently shown options (e.g., "1st one", "2nd one", "1st option")
  → In this case, `should_load_older_memories` depends on whether the option exists in SUGGESTED_STEMS
- **'remove'**: User wants to REMOVE an element (e.g., "remove the bass", "remove the rhythm")
  → In this case, `should_load_older_memories` should be FALSE
- **'other'**: Other types of requests (new song, new section, section switching, etc.)

**STEP 1: Intent Analysis**
- First, analyze the NEW user prompt by comparing it against the `PREVIOUS_USER_PROMPT` and `INTENT_PROMPT`.
- **MANDATORY**: Always examine `SUGGESTED_STEMS` and `MIX_STEMS` to understand the current context.
- **CRITICAL**: When user mentions specific instruments or numbered options, YOU MUST verify they actually exist in the current context.
- **SECTION IDENTIFICATION**: When the user mentions a specific section (e.g.,  "Chorus", "Verse", "Intro", "Outro"), examine `MIX_STEMS` to find the matching section:
    * `MIX_STEMS` is a list of sections, each with `section_name`, `section_role`, and `stems`.
    * `WORKING_SECTION_INDEX` indicates the section that was being worked on in the previous turn.
    * **Decision Rules for `target_working_section_index`:**
        1. **If user explicitly mentions a specific section** (e.g., "코러스 부분 수정", "Verse 작업하고 싶어"):
           - Match user's section reference to `section_role` in `MIX_STEMS`
           - Set `target_working_section_index` to the index of the matched section (0-based)
        2. **If user wants to continue previous work** (e.g., "아까 작업하던 섹션 이어가고 싶어", "새로 생성한 섹션 작업 계속"):
           - Set `target_working_section_index` to `WORKING_SECTION_INDEX`
        3. **If user makes a general request without section specification** (e.g., "드럼 추가해줘", "베이스 바꿔줘"):
           - Set `target_working_section_index` to `WORKING_SECTION_INDEX` (continue on the same section)
        4. **If starting completely new work** (new song, new section):
           - Set `target_working_section_index` to `null` (None)
    * Example: User says "코러스 부분 수정하고 싶어" and MIX_STEMS has section at index 1 with section_role "Chorus / Drop" → set `target_working_section_index` to 1.
    * Example: User says "드럼 바꿔줘" without section mention and WORKING_SECTION_INDEX is 2 → set `target_working_section_index` to 2.
- Your primary goal is to understand if the user is:
    a) Starting a completely new song.
    b) Selecting one of the `SUGGESTED_STEMS` (generation-time candidates) - BUT ONLY if the requested type actually exists.
    c) Replacing/changing something in `MIX_STEMS` with an option from `SUGGESTED_STEMS` (this counts as using suggested stems).
    d) Modifying something in `MIX_STEMS` (currently selected stems) without using suggested options.
    e) Referencing an older context (before the most recent one).
    f) Requesting a new section.

**STEP 2: Flag Determination**
- Based on your analysis, set the boolean flags according to the following rules. All flags default to `false`.
- **IMPORTANT**: These flags are NOT mutually exclusive. Multiple flags can be `true` simultaneously (e.g., both `should_load_older_memories` and `is_use_suggested_stems` can be `true`).

* **1. `is_start_from_scratch`**
    - **Set to `true` if**: The user requests a completely new song, unrelated to the recent context.
    - **Note**: If this is `true`, all other flags are likely `false`.

* **2. `should_load_older_memories`**
    - **[CRITICAL] This flag is DIRECTLY determined by `request_type` and context:**
      
      **If `request_type` is 'add':**
      - **ALWAYS set to `false`** - Adding new elements uses current context only
      - Example: "please add a clap sound" → `false` (generate new clap sound from current context)
      
      **If `request_type` is 'remove':**
      - **ALWAYS set to `false`** - Removing elements only needs current context
      
      **If `request_type` is 'select' or 'replace':**
      - **MANDATORY VALIDATION PROCESS - MUST FOLLOW EXACTLY:**
        
        **Step A: Instrument/Content Type Matching**
        1. Extract the instrument/content type from user's request (e.g. "guitar","bass", etc.)
        2. Extract the numbered reference if any (e.g. "4th", "2nd", "option 1", etc.)
        3. Carefully examine ALL items in `SUGGESTED_STEMS` - read each description fully
        4. Check if ANY item in `SUGGESTED_STEMS` matches the requested instrument/content type
        
        **Step B: Availability Check**
        - **If the requested instrument/content type is NOT found in `SUGGESTED_STEMS`**: 
          → **MUST set to `true`** (need older memories)
        - **If the requested instrument/content type IS found in `SUGGESTED_STEMS`**:
          → Check if the specific numbered option exists
          → If numbered option doesn't exist: set to `true`
          → If numbered option exists: set to `false`
        
        **Step C: Examples for clarity:**
        - User requests "4th guitar" but `SUGGESTED_STEMS` only contains fx/effects → `true`
        - User requests "2nd drum" but `SUGGESTED_STEMS` only has 1 drum option → `true`  
        - User requests "3rd bass" and `SUGGESTED_STEMS` has 4 bass options → `false`
      
      **If `request_type` is 'other':**
      - **Special case for SECTION SWITCHING** (e.g., "continue working on the Verse I created earlier", "아까전에 새로 생성한 Verse 파트 작업을 계속 이어가고 싶어"):
        → **ALWAYS set to `false`** - Switching to a recently created section only requires changing the working section index in the current memory
        → The recently created section already exists in current memory's `chosen_sections`
        → **KEY DISTINCTION**: "Recently created section" (in current session) vs. "reverting to an old version" (needs older memories)
      
      **Additional cases that ALWAYS require `true`:**
      - Explicitly referencing a DIFFERENT version or state from the past ("back to the rock version I made earlier", "go back to the version before I added the guitar")
      - Requesting specific stems that were mentioned in conversation but not in current context
      - **IMPORTANT**: Distinguish between "recently created" (in current memory, set to `false`) vs. "made earlier/before" (in older memory, set to `true`)

* **3. `is_use_suggested_stems`**
    - **Set to `true` if**: The user selects a specific option from a set of suggestions OR wants to replace/change an existing stem with a different option. This includes:
      - Selecting numbered options ("1st", "option 2", "2nd sound")
      - Replacing existing stems ("guitar instead of 2nd", "change to something else")
      - Choosing from descriptions or specific alternatives
    - **CRITICAL**: This should be `true` REGARDLESS of whether `should_load_older_memories` is `true` or `false`
    - **Key insight**: If the user mentions any numbered option or wants to select/replace with a specific choice, this flag should be `true`
    - **Example**: "guitar sound instead of 2nd sound that I chose earlier" → Both `should_load_older_memories=true` AND `is_use_suggested_stems=true`

* **4. `is_start_new_branch`**
    - **Set to `true` ONLY IF**: `should_load_older_memories` is `true` AND the user intends to create a new version or fork from that older point in time.

* **5. `is_start_new_section`**
    - **Set to `true` if**: The user explicitly asks to add a new structural part to the song (e.g., "add a chorus", "make a next section", "extend this part").

* **6. `is_publish_song`**
    - **Set to `true` if**: The user explicitly asks to publish/post the song (e.g., "publish/post the song", "finish the song").

**STEP 3: Final Details**
- **`target_working_section_index`**: Set this to the index of the section the user wants to work on:
    * **If user explicitly mentions a specific section**: Find the matching section in `MIX_STEMS` by its `section_role` and set this to its index (0-based).
    * **If user wants to continue previous work or makes a general request**: Set this to `WORKING_SECTION_INDEX` (the section currently being worked on).
    * **If starting completely new work** (new song, new section): Set this to `null` (None).
    * This is critical for section-specific modifications.
"""


# - **`reasoning`**: Provide a detailed explanation including: 1) Request type classification and why, 2) What the user requested, 3) What you found in SUGGESTED_STEMS/MIX_STEMS, 4) Whether the requested content exists in current context, 5) Your decision rationale for EACH flag, 6) Section identification logic (explicit mention vs. continuing previous work vs. new work), 7) The value of WORKING_SECTION_INDEX and target_working_section_index, 8) Why multiple flags might be true simultaneously if applicable.


def build_strategy_user_msg(
    user_prompt: str,
    previous_context: str = "",
    intent_focused_prompt: str = "",
    mix_lines: list = [],
    suggested_lines: list = [],
    working_section_index: int | None = None,
) -> str:
    return f"""
NEW user prompt:
{user_prompt}

LAST MEMORY SNAPSHOT:
[INTENT_PROMPT]
{intent_focused_prompt}

[PREVIOUS_USER_PROMPT]
{previous_context}

[MIX_STEMS]
{json.dumps(mix_lines, indent=4) if mix_lines else "- none"}


[SUGGESTED_STEMS]
{chr(10).join(suggested_lines) if suggested_lines else "- none"}

[WORKING_SECTION_INDEX]
{working_section_index}
"""


def build_strategy_openai_function():
    return {
        "name": "decide_strategy_for_recent",
        "description": "Decide strategy flags for using the last memory and whether to include older ones.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_type": {
                    "type": "string",
                    "enum": ["add", "replace", "select", "remove", "other"],
                    "description": "Type of user request: 'add' (adding new element), 'replace' (changing existing element), 'select' (choosing from options), 'remove' (deleting element), or 'other' (new song, new section, section switching, etc.).",
                },
                "intent_focused_prompt": {
                    "type": "string",
                    "description": "The user's intent, as determined by context.",
                },
                "should_load_older_memories": {
                    "type": "boolean",
                    "description": "True if the user's request cannot be fulfilled by current SUGGESTED_STEMS or MIX_STEMS; false if current context has all needed information. For 'add' or 'remove' requests, this should ALWAYS be false. For 'other' type requests about recently created sections, this should also be false.",
                },
                "is_start_from_scratch": {
                    "type": "boolean",
                    "description": "True if the user wants to start a completely new and different song.",
                },
                "is_use_suggested_stems": {
                    "type": "boolean",
                    "description": "True if the user wants to select or replace with an option among SUGGESTED_STEMS (generation-time candidates). This includes both new selections and replacements of existing stems.",
                },
                "is_start_new_branch": {
                    "type": "boolean",
                    "description": "True if the user wants to start a NEW song branching from a specific earlier point/context of the previous work.",
                },
                "is_start_new_section": {
                    "type": "boolean",
                    "description": "True if the user wants to generate and add a new section.",
                },
                "is_publish_song": {
                    "type": "boolean",
                    "description": "True if the user wants to publish the song.",
                },
                # "reasoning": {
                #     "type": "string",
                #     "description": "Reasoning process for the decision.",
                # },
                "target_working_section_index": {
                    "type": ["integer", "null"],
                    "description": "The 0-based index of the section to work on. Rules: 1) If user explicitly mentions a specific section (e.g., 'Chorus', 'Verse', '코러스'), find it in MIX_STEMS by section_role and return its index. 2) If user wants to continue previous work or makes a general request without section specification, return WORKING_SECTION_INDEX. 3) If starting completely new work (new song, new section), return null.",
                },
            },
            "required": [
                "intent_focused_prompt",
                "should_load_older_memories",
                "is_start_from_scratch",
                "is_use_suggested_stems",
                "is_start_new_branch",
                "is_start_new_section",
                "is_publish_song",
                # "reasoning",
                "target_working_section_index",
            ],
        },
    }


def build_best_memory_context_summary(candidates: list[MemoryData]) -> list[str]:
    context_summaries: list[str] = []
    for idx_candidate, candidate in enumerate(candidates):
        memory_id = candidate.memory_id.split("_")[-1].replace(".json", "")
        user_prompt_prev = candidate.user_prompt
        intent_prompt_prev = candidate.intent_focused_prompt
        if candidate.chosen_sections:
            mix_lines = [
                f"- {s.category}: {s.caption} | {s.uri}"
                for s in candidate.chosen_sections[candidate.working_section_index]
            ]
        else:
            mix_lines = ["None"]
        if candidate.generated_stems:
            suggested_lines = [
                f"- {idx+1}. {s.category}: {s.caption} | {s.uri}"
                for idx, s in enumerate(candidate.generated_stems)
            ]
        else:
            suggested_lines = ["None"]

        summary = (
            f"** MEMORY INDEX NUMBER: {idx_candidate+1} **\n"
            f"[MEMORY ID]: {memory_id}\n"
            f"[USER PROMPT]: {user_prompt_prev}\n\n"
            f"[INTENT PROMPT]: {intent_prompt_prev}\n\n"
            "[MIX_STEMS]: " + ("\n".join(mix_lines) if mix_lines else "- none") + "\n\n"
            "[SUGGESTED_STEMS]: "
            + ("\n".join(suggested_lines) if suggested_lines else "- none")
        )

        context_summaries.append(summary)
    context_summaries_str = "\n".join(context_summaries)
    return context_summaries_str


def build_best_memory_system_msg():
    return """
[Role & Goal]
You are a specialized AI tool, a "Memory Selector". Your single task is to analyze a user's request and select the single best OLDER memory context from a provided list. Your output must be only the selected `MEMORY ID` (or file name).

[Absolute Rules]
1.  **SELECT AN OLDER MEMORY**: You MUST select a memory with `MEMORY_INDEX_NUMBER >= 2`. You are strictly forbidden from selecting the most recent memory (index 1).
2.  **CHRONOLOGY IS KEY**: `CANDIDATE_MEMORY_CONTEXTS` are listed from most-recent (index 1) to oldest (index 2, 3, ...). Use this order to resolve time-based user requests.
3.  **SELECTION PHRASES**: User requests like `"~를 선택해줘"`, `"1번째"`, `"option 2"` refer to the 1-based index within a candidate's `SUGGESTED_STEMS` list. Your job is to find which candidate's list they are referring to.
4.  **STEM URI DETECTION**: If the user's request contains a stem URI in the format `<memory_id>-<category>-<num>.aac` (e.g., "I want to use 94233fb0-e492-45a2-94d1-7f293cb665fd-rhythm-0.aac"), extract the `<memory_id>` portion and select the memory with that exact MEMORY ID. This rule takes precedence over all other selection logic.

[Evaluation Workflow]
To find the best older memory, follow this exact procedure:

**STEP 0: Check for Stem URI (Priority Check)**
- First, scan the user's request for any stem URI pattern matching `<memory_id>-<category>-<num>.aac`.
- If found, extract the `<memory_id>` portion and immediately select the memory with that MEMORY ID.
- If a stem URI is detected, skip all remaining steps and return that MEMORY ID.

**STEP 1: Iterate Through Older Candidates**
- If no stem URI was detected, evaluate EACH candidate memory `k` where `MEMORY_INDEX_NUMBER` is 2 or greater.

**STEP 2: Gather Evidence for Each Candidate `k`**
- For each `k`, gather and analyze key pieces of evidence:
    * **Primary Evidence (from `k`)**: Examine the `SUGGESTED_STEMS` list of candidate `k`. This is your most important signal.
        * Does it contain the instrument, style, or numeric option (e.g., "2nd one") mentioned in the new user prompt?
        * For requests to *replace* an existing stem (e.g., "replace the bass with another one"), you must verify that candidate `k`'s `MIX_STEMS` contains 'bass' and its `SUGGESTED_STEMS` offers alternative bass options.
    * **Supporting Evidence (from `k-1`)**: Examine the `INTENT_PROMPT` of the memory just after `k` (i.e., memory `k-1`). This can provide context for what might have been chosen from `k`'s suggestions. Treat this as optional, supporting information.

**STEP 3: Score and Select**
- Score each candidate `k` based on how well the user's prompt matches the **Primary Evidence**, with extra points if the **Supporting Evidence** also aligns.
- After evaluating all candidates (with index >= 2), select the one with the highest score.

**STEP 4: Tie-Breaking Rule**
- If multiple older candidates have a high score, choose the most recent one among them (the one with the smallest `MEMORY_INDEX_NUMBER`).

**[Critical Evaluation Nuance]**
- **A user may refer back to the options at time `k` and choose a *different* option than they did originally.** For example, if they originally chose option 1 from time `k`, they might now say "I'll try option 2". In this case, you must still select candidate `k` as the correct context.

**STEP 5: Final Output**
- Return ONLY the `MEMORY ID` or file name of the single best candidate you have selected. If no older candidate is a plausible match, you must still choose the best-fitting one among the older options rather than defaulting to the most recent one.
"""


def build_best_memory_user_msg(
    user_prompt: str, intent_focused_prompt: str, context_summaries: list[str]
) -> str:
    return f"""
    Given the following user prompt, inferred user intent, and 'CANDIDATE MEMORY CONTEXTS', choose the SINGLE best memory file to use. 
    Carefully analyze which candidate's SUGGESTED_STEMS align with the user's selection request (if any), and ensure MIX_STEMS reflects already-selected stems mentioned in the request.
    * [User prompt]: {user_prompt}\n
    * [User intent]: {intent_focused_prompt}\n
    * [CANDIDATE MEMORY CONTEXTS]:
    {chr(10).join(context_summaries)}
    """


def build_best_memory_openai_function():
    return {
        "name": "choose_memory_strategy",
        "description": "Select the best prior memory file for the user's new request and intent by analyzing MIX_STEMS (already chosen) and SUGGESTED_STEMS (options shown).",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_previous_memory": {
                    "type": "string",
                    "description": "After analyzing MIX_STEMS and SUGGESTED_STEMS in each candidate for the given user prompt and user intent, select the memory file that you think is most appropriate.",
                },
            },
            "required": [
                "selected_previous_memory",
            ],
        },
    }
