import json
import os
import pathlib
from datetime import datetime  # noqa: F401
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


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
            "previous_context": req_data.get("context", {}).get("previousContext", []),
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
        result = json.loads(response.choices[0].message.content)
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
        return response.choices[0].message.content.strip()
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
