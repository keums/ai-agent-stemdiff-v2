import json
import os
import pathlib
from datetime import datetime  # noqa: F401
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

from models import IntentResult, MemoryData

# Load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def build_intent_router_system_msg():
    return """
You are an intent classification agent for a music creation chatbot. Your task is to understand the TRUE INTENT behind each user message by analyzing what the user ultimately wants to accomplish, not just matching keywords.

## Classification Principles

**Identify the PRIMARY action the user wants to take RIGHT NOW:**

1. **chat** - User's primary goal is to obtain information, explanation, or have a conversation
   - Asking questions about concepts, styles, or how things work
   - Seeking clarification or explanation before taking action
   - Social responses (greetings, thanks, acknowledgments)
   - Examples: "What is jazz?", "Explain alternative rock to me", "How does this work?"
   - **CRITICAL**: Even if user mentions future creation intent ("I want to make X, but what is X?"), if they're asking for explanation first, it's CHAT

2. **select** - User wants to choose one option from previously presented alternatives
   - References to specific choices, numbers, or ordinal positions
   - Implies that options were already given in conversation history
   - Examples: "the first one", "number 3", "I'll pick this", "use that stem"

3. **generate** - User wants to create NEW music content
   - Direct creation requests without referencing existing elements
   - Requesting new stems, instruments, or musical elements
   - Examples: "create a beat", "add drums", "make jazz music"
   - Note: NOT generate if user is asking about it first (see chat)

4. **remix** - User wants to modify existing music holistically
   - Changing the overall arrangement, style, or feel of current mix
   - Examples: "remix this", "change the vibe", "make it faster"

5. **remove** - User wants to delete specific audio elements
   - Eliminating existing stems from the current mix
   - Examples: "remove the bass", "delete this stem", "take out drums"

6. **replace** - User wants to swap one audio element with another
   - Substituting an existing stem while keeping others
   - Examples: "replace drums with percussion", "swap this for that"

7. **post** - User indicates work completion and wants to finalize/publish
   - Signals the end of the creative session
   - Examples: "done", "publish this", "finalize the track"

## Classification Strategy

1. Read the ENTIRE user message to understand context
2. Identify what the user wants to happen NEXT (not eventually)
3. If multiple intents exist, prioritize by: post > remove/replace > remix > select > generate > chat
4. However, if the PRIMARY sentence is a question seeking information, classify as CHAT regardless of other mentions

## Example Reasoning

- "I want to make alternative rock. What is alternative rock?" 
  → PRIMARY: asking for explanation → CHAT

- "What is jazz? I want to create it."
  → PRIMARY: asking for explanation → CHAT

- "Create alternative rock music"
  → PRIMARY: direct creation request → GENERATE

- "I'll go with the second option"
  → PRIMARY: choosing from options → SELECT

## Language Detection Rule (Applies to ALL outputs)

**CRITICAL**: Detect the user's PRIMARY language by analyzing BOTH Intent History + Current Prompt together:
- Look at the overall conversation pattern in Intent History
- If most of the conversation has been in Korean, use Korean - even if the current prompt is "okay", "yes", or other short English words
- If most of the conversation has been in English, use English
- If Intent History is empty, follow the language of the current prompt
- Examples:
  * Intent History (Korean dominant) + Current: "okay" → Respond in KOREAN
  * Intent History (Korean dominant) + Current: "첫번째" → Respond in KOREAN  
  * Intent History (English dominant) + Current: "first one" → Respond in ENGLISH
  * Intent History (empty) + Current: "얼터너티브 록이 뭐야?" → Respond in KOREAN

## Output Requirements

You must provide exactly 3 fields:

1. **request_type**: One of [chat, select, generate, remix, remove, replace, post]

2. **intent_focused_prompt**: 
   - Restate the user's request clearly with full conversation context
   - Make ambiguous references explicit using context from history
   - Examples:
     * "첫번째" → "세 가지 재즈 드럼 옵션 중 첫 번째를 선택하고 싶어"
     * "first one" → "I want to select the first jazz drum option from three choices"
     * "okay" (when agreeing to add bass) → "네, 제안된 베이스 스템을 현재 믹스에 추가하고 싶어"
   
3. **response**: 
   - If request_type is "chat": Provide a helpful, informative answer
   - Otherwise: Empty string ""
"""


async def intent_agent_router(
    user_prompt: str, memory_data: MemoryData
) -> IntentResult:
    """
    Intent Agent is responsible for determining the type of user request based on the user's prompt.
    Uses modern structured output with JSON Schema.

    Returns:
        IntentResult object with three fields:
        - request_type: One of (chat, remix, generate, select, remove, replace, post)
        - intent_focused_prompt: Clarified version of user's request with full context
        - response: Chat response (non-empty only when request_type is 'chat')

        Example:
            IntentResult(
                request_type="chat",
                intent_focused_prompt="User is asking what the chatbot can do",
                response="I can help you create music..."
            )
            IntentResult(
                request_type="generate",
                intent_focused_prompt="User wants to create rock music",
                response=""
            )
    """
    system_prompt = build_intent_router_system_msg()

    intent_history = memory_data.intent_history
    last_agent_response = memory_data.last_agent_response
    user_prompt_for_llm = f"""
    Here is the user's prompt and intent history:
    1.  **The User's Prompt:** "{user_prompt}"
    2.  **Intent History:** {intent_history} # [most old request, ..., most recent request] sorted in order.
    3.  **Last Agent Response:** {last_agent_response}
    """

    # Define JSON Schema for structured output
    intent_schema = {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "enum": [
                    "chat",
                    "remix",
                    "generate",
                    "select",
                    "remove",
                    "replace",
                    "post",
                ],
                "description": "The type of user request",
            },
            "intent_focused_prompt": {
                "type": "string",
                "description": "A clarified version of the user's request with full context from conversation history. Always required.",
            },
            "response": {
                "type": "string",
                "description": "Response message for chat requests. Should contain the chat response when request_type is 'chat', empty string otherwise.",
            },
        },
        "required": ["request_type", "intent_focused_prompt", "response"],
        "additionalProperties": False,
    }

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_llm},
            ],
            temperature=0.5,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intent_classification",
                    "strict": True,
                    "schema": intent_schema,
                },
            },
        )
        content = response.choices[0].message.content
        if content is None:
            return IntentResult(
                request_type="chat",
                intent_focused_prompt=user_prompt,
                response="I'm here to help you create music. What would you like to do?",
            )
        else:
            result = json.loads(content)
            request_type = result.get("request_type", "chat")
            intent_focused_prompt = result.get("intent_focused_prompt", user_prompt)
            response_text = result.get("response", "")

            # If it's chat but response is empty, provide default
            if request_type == "chat" and not response_text:
                response_text = (
                    "I'm here to help you create music. What would you like to do?"
                )

            return IntentResult(
                request_type=request_type,
                intent_focused_prompt=intent_focused_prompt,
                response=response_text,
            )
    except Exception as e:
        print(f"Intent Agent LLM call failed: {e}")
        return IntentResult(
            request_type="chat",
            intent_focused_prompt=user_prompt,
            response="I'm here to help you create music. What would you like to do?",
        )


# ------------------------------------------------------------ #
def summarize_schema_for_llm(file_path: str) -> Dict[str, Any] | None:
    """
    Reads a data_schema JSON file and extracts a concise summary.
    Returns None if the file has an error status or is invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("request"):
            return None

        req_data = data["request"][0]
        answer = req_data.get("answers", [{}])[0]

        # Skip files with an error status in the answer
        if answer.get("status") == "error":
            print(f"Skipping error file: {os.path.basename(file_path)}")
            return None

        summary = {
            "file_name": os.path.basename(file_path),
            "previous_user_prompt": req_data.get("chatMessage", {}).get("chatText"),
            "intent_focused_prompt": req_data.get("chatMessage", {}).get(
                "intentFocusedPrompt"
            ),
            "intent_history": req_data.get("intentHistory", []),
            "current_mix_stems": [
                {
                    "category": stem.get("category"),
                    "instrumentName": stem.get("instrumentName", ""),
                    "sectionRole": stem.get("sectionRole"),
                    "caption_preview": stem.get("caption", "")[:100] + "...",
                }
                for stem in answer.get("mix", {}).get("mixData", {}).get("stems", [])
            ],
            "suggested_stems": [
                {
                    "category": stem.get("category"),
                    "instrumentName": stem.get("instrumentName", ""),
                    "sectionRole": stem.get("sectionRole"),
                    "caption_preview": stem.get("caption", "")[:100] + "...",
                }
                for stem in answer.get("suggestedStems", [])
            ],
        }
        return summary
    except (IOError, json.JSONDecodeError, IndexError) as e:
        print(f"Could not process file {file_path}: {e}")
        return None


def validate_latest_memory_context(
    new_user_prompt: str, latest_summary: Dict[str, Any]
) -> bool:
    """
    Uses a fast LLM call to validate if the latest file's context is sufficient,
    given the new user prompt.
    """

    system_prompt = """
    You are a high-speed context validator in a music creation chatbot. Your task is to look at the user's NEW prompt and the summary of the LAST interaction, and determine if the last interaction's context is sufficient to proceed. The context is sufficient unless the new prompt explicitly references a much older idea or seems completely disconnected.

    Respond with ONLY a JSON object: {"is_sufficient": true} or {"is_sufficient": false}.
    - Use `true` if the new prompt is a direct continuation (e.g., selecting an option, asking for the next step).
    - Use `false` if the new prompt says "let's go back to the first idea" or "I liked the vibe from two steps ago", indicating a need to search history.

    Domain-specific instruction (decide using natural language understanding only):
    - If the NEW prompt explicitly asks to work with a particular stem type (e.g., "다른 리듬", "I want a different rhythm", "add bass", "make a new lead") and that stem type is not represented or was not the focus in the LAST interaction's `current_mix_stems` or `suggested_stems`, then the LAST file is NOT sufficient. In that case, return {"is_sufficient": false} so we search history (e.g., find where that category was first proposed).
    """

    user_prompt_for_llm = f"""
    Here is the data:
    1.  **The User's NEW Request:** "{new_user_prompt}"
    2.  **Summary of the LAST Interaction:** {json.dumps(latest_summary, indent=2, ensure_ascii=False)}

    Based on the user's NEW request, is the context from the LAST interaction sufficient to continue, or do we need to search the history?
    Respond with JSON: {{"is_sufficient": boolean}}
    """

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_llm},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            return False
        result = json.loads(content)
        return result.get("is_sufficient", False)
    except Exception as e:
        print(f"Validator LLM call failed: {e}")
        return False


def select_best_memory_from_history(
    new_user_prompt: str, summaries: List[Dict[str, Any]]
) -> str:
    """
    Uses a powerful LLM to select the best file from a list of historical summaries,
    based on the new user prompt.
    """
    system_prompt = """
    You are an expert AI agent in a music generation pipeline. Your task is to analyze a user's new request and summaries of several recent session files. Select the SINGLE most relevant file from the history to use as context for satisfying the user's new request.
    Your response MUST be ONLY the filename of your choice, and nothing else.

    Selection guidelines (use natural language understanding):
    - If the user asks for a particular stem type (rhythm, low, mid, high, fx) or instrument name (e.g., a different rhythm), prefer the most recent file in which that stem type was first proposed or is clearly the focus (either in `current_mix_stems` or `suggested_stems`).
    - Avoid picking the latest file if its stems focus on a different category than the one requested by the user.
    - When in doubt, choose the file whose intent and stems best match the user's explicit request.
    """

    user_prompt_for_llm = f"""
    Here is the data:
    1.  **The User's NEW Request:** "{new_user_prompt}"
    2.  **Summaries of Recent Interactions (newest to oldest):** {json.dumps(summaries, indent=2, ensure_ascii=False)}

    Given the user's NEW request, which of the historical files provides the most relevant context to proceed?

    Provide ONLY the filename of your chosen file.
    """

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_llm},
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        print(f"Selector LLM call failed: {e}")
        return summaries[0]["file_name"] if summaries else ""


def intelligent_schema_selector(
    directory: str, new_user_prompt: str, num_deep_search: int = 5
) -> str:
    """
    Implements a two-step process to efficiently select the most relevant schema file
    based on a new user prompt.
    """
    print(os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
        )

    try:
        all_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.startswith("data_schema_") and f.endswith(".json")
        ]
        all_files.sort(key=os.path.getmtime, reverse=True)
    except FileNotFoundError:
        return f"Error: Directory not found at {directory}"

    if not all_files:
        return "No data_schema files found."

    # --- Step 1: Fast Path ---
    latest_file_path = all_files[0]
    latest_summary = summarize_schema_for_llm(latest_file_path)

    if latest_summary:
        print(
            f"Fast Path: Validating the latest file '{os.path.basename(latest_file_path)}' against the new user prompt..."
        )
        if validate_latest_memory_context(new_user_prompt, latest_summary):
            print("Validation successful. The latest file is sufficient.")
            return os.path.basename(latest_file_path)

    # --- Step 2: Deep Search (Fallback) ---
    print("Validation failed or latest file is invalid. Initiating Deep Search...")

    summaries = []
    for f in all_files[:num_deep_search]:
        summary = summarize_schema_for_llm(f)
        if summary:
            summaries.append(summary)

    if not summaries:
        return "No valid files found for deep search."

    return select_best_memory_from_history(new_user_prompt, summaries)


if __name__ == "__main__":
    # --- Setup for Demonstration ---
    DATA_DIR = "../output/data_schema"

    # --- SIMULATION ---
    print("--- Simulation 1: Simple Continuation ---")
    # The user makes a simple choice, so the latest file should be sufficient.
    new_prompt_1 = "아까 선택했던 첫번째 리듬 스템 말고 나머지 스템을 사용하고 싶어"
    chosen_file_1 = intelligent_schema_selector(DATA_DIR, new_prompt_1)
    print(f"\nChosen file by the agent: {chosen_file_1}")

    print("\n" + "=" * 50 + "\n")
