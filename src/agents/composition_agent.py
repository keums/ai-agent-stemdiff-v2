import json
import os
import pathlib
from datetime import datetime  # noqa: F401

import openai
from dotenv import load_dotenv

from models import CompositionAgentOutput, MemoryData

# Load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def build_composition_agent_system_msg():
    return ""


async def composition_agent(
    user_prompt: str, memory_data: MemoryData
) -> CompositionAgentOutput:
    """
    Composition Agent is responsible for determining the type of composition action to take based on the user's prompt.

    Returns:
        CompositionAgentOutput object with the following fields:
        - start_from_scratch: bool
        - start_new_branch: bool
        - start_new_section: bool
        - add_new_stem: bool
    """
    system_prompt = build_composition_agent_system_msg()

    user_prompt_for_llm = f"""
    Here is the user's prompt:
    1.  **The User's Prompt:** "{user_prompt}"
    """

    # Define JSON Schema for structured output
    intent_schema = {
        "type": "object",
        "properties": {
            "start_from_scratch": {
                "type": "boolean",
                "description": "Whether to start composing from scratch",
            },
            "start_new_section": {
                "type": "boolean",
                "description": "Whether to start a new section in the composition",
            },
            "add_new_stem": {
                "type": "boolean",
                "description": "Whether to add a new stem to the existing composition",
            },
            "continue_previous_stem": {
                "type": "boolean",
                "description": "Whether to continue the previous stem in the composition",
            },
        },
        "required": [
            "start_from_scratch",
            "start_new_section",
            "add_new_stem",
            "continue_previous_stem",
        ],
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
                    "name": "composition_action_classification",
                    "strict": True,
                    "schema": intent_schema,
                },
            },
        )
        content = response.choices[0].message.content
        if content is None:
            return CompositionAgentOutput(
                start_from_scratch=False,
                start_new_branch=False,
                start_new_section=False,
                add_new_stem=False,
            )
        else:
            result = json.loads(content)
            return CompositionAgentOutput(
                start_from_scratch=result.get("start_from_scratch", False),
                start_new_branch=result.get("start_new_branch", False),
                start_new_section=result.get("start_new_section", False),
                add_new_stem=result.get("add_new_stem", False),
            )
    except Exception as e:
        print(f"Composition Agent LLM call failed: {e}")
        return CompositionAgentOutput(
            start_from_scratch=False,
            start_new_branch=False,
            start_new_section=False,
            add_new_stem=False,
        )
