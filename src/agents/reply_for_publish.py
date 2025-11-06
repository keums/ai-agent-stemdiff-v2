import os
from typing import Any, Dict, List, Optional

import anthropic
import openai
from pydantic import BaseModel, Field

from models import ContextSong, Stem
from tools.mcp_base import tool

CLAUDE_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 2000
TEMPERATURE = 1
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # claude, openai


# GlobalInformationGeneration 입출력 스키마
class ReplyForPublishSongInput(BaseModel):
    """글로벌 음악 정보 생성 도구에 대한 입력 스키마"""

    user_prompt: str = Field(..., description="User input text")
    intent_focused_prompt: str = Field(..., description="User input text")
    mix_stem_diff: List[List[Dict[str, Any]]] = Field(
        ..., description="List of mix stem diff URIs with metadata"
    )
    context_song_info: Optional[Dict[str, Any]] = Field(
        ..., description="Context song info"
    )
    caption_input: Optional[str] = Field(
        default=None, description="Description of the song from caption model"
    )


class ReplyForPublishSongOutput(BaseModel):
    """글로벌 음악 정보 생성 도구에 대한 출력 스키마"""

    reply: str = Field(
        description="Comprehensive reply explaining what was requested, what will be generated, and asking for user selection if needed"
    )
    genre: List[str] = Field(description="Genre of the stems")
    mood: List[str] = Field(description="Mood of the stems")
    instruments: List[str] = Field(description="Main instruments used in the stems")
    title: str = Field(description="Creative title for the stems")
    bpm: int = Field(description="BPM of the stems")
    key: str = Field(description="Musical key of the stems")
    music_caption: str = Field(description="User-friendly description of the song")
    thumbnail_url: str = Field(description="Thumbnail URL of the song")
    # reasoning: str = Field(description="Reasoning for the reply")

    def to_dict(self):
        return {
            "reply": self.reply,
            "genre": self.genre,
            "mood": self.mood,
            "instruments": self.instruments,
            "title": self.title,
            "music_caption": self.music_caption,
            "bpm": self.bpm,
            "key": self.key,
            "thumbnail_url": self.thumbnail_url,
        }


tool_schema = {
    "name": "reply_for_publish_song",
    "description": "Generate a comprehensive reply explaining the music generation process and results",
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "Responding to user posting requests",
            },
            "genre": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Genre of this specific track",
            },
            "mood": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mood of this specific track",
            },
            "instruments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main instruments used in this track",
            },
            "title": {
                "type": "string",
                "description": "Creative title for this specific track",
            },
            "bpm": {
                "type": "integer",
                "description": "BPM of this track",
            },
            "key": {
                "type": "string",
                "description": "Musical key of this track",
            },
            "music_caption": {
                "type": "string",
                "description": "User-friendly description of this specific track. Please use the same language as the user_language.",
            },
        },
        "required": [
            "reply",
            "genre",
            "mood",
            "instruments",
            "title",
            "bpm",
            "key",
            "music_caption",
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
        tool_choice={"type": "tool", "name": "generate_reply"},
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

    # messages 처리
    for msg in messages:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=openai_messages,
        functions=[openai_function],
        function_call={"name": "reply_for_publish_song"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response


async def call_llm_for_music_info(
    system_message, messages, llm_provider=DEFAULT_LLM_PROVIDER
):
    """LLM 호출 통합 함수"""
    if llm_provider.lower() == "openai":
        # print("🚀🚀🚀 Start: call_openai_api (generate_music_info) 🚀🚀🚀")
        return await call_openai_api(system_message, messages)
    elif llm_provider.lower() == "claude":
        # print("🚀🚀🚀 Start: call_claude_api (generate_music_info) 🚀🚀🚀")
        return await call_claude_api(system_message, messages)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


@tool(
    name="reply_for_publish_song",
    description="Agent orchestrator for reply generation",
    input_schema=ReplyForPublishSongInput,
    output_schema=ReplyForPublishSongOutput,
)
async def reply_for_publish_song(
    user_prompt: str,
    intent_focused_prompt: str,
    mix_stem_diff: List[List[Stem]],
    context_song_info: ContextSong,
    caption_input: Optional[str],
) -> ReplyForPublishSongOutput:
    """Reply orchestrator tool"""

    arranged_sections_order = (
        context_song_info.arranged_sections_order if context_song_info else []
    )
    bpm = context_song_info.bpm if context_song_info else 0
    key = context_song_info.key if context_song_info else ""
    # unique한 (instrumentName, category) 페어만 수집
    unique_pairs = set()
    mix_stem_diff_for_llm = []

    for section in mix_stem_diff:
        for stem in section:
            category = stem.category
            instrument_name = stem.instrument_name
            pair = (instrument_name, category)

            # 아직 추가되지 않은 unique한 페어만 추가
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                mix_stem_diff_for_llm.append(
                    {
                        "category": category,
                        "instrumentName": instrument_name,
                        "caption": stem.caption,
                    }
                )

    system_message = """\n
    You are a music AI assistant that provides engaging summaries of music generation results.
    Focus on the generated music caption as the main description, and connect it with the user's original request to create a cohesive explanation of what was created for them.

    After providing the summary and explanation, you may also offer additional suggestions or improvements for user request and if you identify areas that could be enhanced or expanded upon.
    If there is a next request, please mention specifically what request remains and ask if the user wants to continue processing it.
    Use proper emoji if possible.
    Please check language of USER_PROMPT and answer according to the input language.

    1. Reply: provide a summary of the generated music based on all the information.
    2. Title: provide a creative song title (around 5 words or less) that captures the essence of the generated music and user's intent. The title should be creative yet meaningful, reflecting the overall content and mood of the music.
    3. Genre/Mood: provide the genre and mood of the generated music by analyzing and summarizing the content above. List the most relevant genres and moods that best describe the music. (up to three genres and moods each)
    4. Instruments: provide the main instruments used in the generated music by analyzing the content above. List the key instruments that are present or would be suitable for this type of music. (up to four instruments)
    5. Music Caption: actively utilize the given caption information to modify and deliver it in a way that is easy for users to understand and clear. Transform the technical or raw caption into user-friendly language that clearly conveys the music's characteristics.
    6. Key: e.g.) C#M => C# major, Dbm => Db minor, etc.
    """

    messages = [
        {
            "role": "user",
            "content": f"""Create a natural and professional response as a music producer collaborating with the user:
* USER_PROMPT: "{user_prompt}" (What user actually said)
* INTENT_FOCUSED_PROMPT: "{intent_focused_prompt}" (The musical direction we're exploring)
* BPM: {bpm} (BPM of the song)
* KEY: {key} (Musical key of the song)
* MIX_STEM_DIFF: {mix_stem_diff_for_llm} (The unique description of instruments and categories of the completed song so far. Use them as a basis for your recommendation.)
* SONG_STRUCTURE: {arranged_sections_order} (Structure of the song)
* CAPTION_INPUT: {caption_input} (Description of the song from caption model)
* IMPORTANT: Please check language of USER_PROMPT and answer according to the input language.
""",
        }
    ]

    response = await call_llm_for_music_info(system_message, messages)

    # OpenAI와 Claude 응답 처리
    if hasattr(response, "choices") and response.choices:
        # OpenAI 응답 처리
        if response.choices[0].message.function_call:
            import json

            function_args = json.loads(
                response.choices[0].message.function_call.arguments
            )

            output = ReplyForPublishSongOutput(
                reply=function_args.get("reply", ""),
                genre=function_args.get("genre", []),
                mood=function_args.get("mood", []),
                instruments=function_args.get("instruments", []),
                title=function_args.get("title", ""),
                bpm=function_args.get("bpm", 0),
                key=function_args.get("key", ""),
                music_caption=function_args.get("music_caption", ""),
                thumbnail_url="",
            )
            print_output(output)
            return output
    elif hasattr(response, "content"):
        # Claude 응답 처리
        for content_block in response.content:
            if content_block.type == "tool_use":
                output = ReplyForPublishSongOutput(
                    reply=content_block.input.get("reply", ""),
                    genre=content_block.input.get("genre", []),
                    mood=content_block.input.get("mood", []),
                    instruments=content_block.input.get("instruments", []),
                    title=content_block.input.get("title", ""),
                    bpm=content_block.input.get("bpm", 0),
                    key=content_block.input.get("key", ""),
                    music_caption=content_block.input.get("caption", ""),
                    thumbnail_url="",
                )
                print_output(output)
                return output

    # 기본 응답 (fallback)
    default_reply = """Hi! I'll create the music you requested! 🎵"""

    output = ReplyForPublishSongOutput(reply=default_reply)
    print_output(output)
    return output


def print_output(output):
    print("\n📊 Reply for publishing: \n", output)
