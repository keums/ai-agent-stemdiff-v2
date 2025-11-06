import os
from typing import Any, Dict, List, Optional

import anthropic
import openai
from pydantic import BaseModel, Field

from models import ContextSong
from tools.mcp_base import tool

CLAUDE_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 3000
TEMPERATURE = 0.6
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # claude, openai


# GlobalInformationGeneration 입출력 스키마
class ReplyAgentInput(BaseModel):
    """글로벌 음악 정보 생성 도구에 대한 입력 스키마"""

    user_prompt: str = Field(..., description="User input text")
    intent_focused_prompt: str = Field(..., description="User input text")
    text_prompts: List[Dict[str, str]] = Field(
        description="List of stem or song prompts for batch embedding",
    )
    prompt_stem_info: List[Dict[str, Any]] = Field(
        default=[], description="List of prompt stem info"
    )
    working_section: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="List of mix stem diff URIs with metadata"
    )
    context_song_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Context song info"
    )
    request_type: Optional[str] = Field(default=None, description="Request type")
    unique_stems_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Unique stems info"
    )


class ReplyAgentOutput(BaseModel):
    """글로벌 음악 정보 생성 도구에 대한 출력 스키마"""

    user_response: str = Field(
        description="Response to user request - summary of what user requested and how it will be processed"
    )
    title: str = Field(
        description="Short and clear title for the answer or suggestion (e.g., 'Generate Guitar', 'Remove Drum')"
    )
    result_description: str = Field(
        description="Content describing the results of the request processing"
    )
    card_title: str = Field(
        description="Title for instrument suggestions in format '{INSTRUMENT_NAME} Suggestion for {CURRENT_SECTION_ROLE}'"
    )
    stem_descriptions: List[str] = Field(
        description="List of short descriptions for generated stems (max 10 words each), ordered by PROMPT_STEM_INFO"
    )
    suggestion: str = Field(
        description="Proactive suggestion for next steps based on current state"
    )
    instrument_name: List[str] = Field(description="Instrument name of the stems")

    @property
    def reply(self) -> str:
        """Legacy compatibility property that combines structured components into a single reply"""
        reply_parts = []

        if self.user_response:
            reply_parts.append(self.user_response)
        if self.result_description:
            reply_parts.append(self.result_description)
        if self.stem_descriptions:
            for i, desc in enumerate(self.stem_descriptions, 1):
                reply_parts.append(f"{i}. {desc}")
        if self.instrument_name:
            for i, name in enumerate(self.instrument_name, 1):
                reply_parts.append(f"{i}. {name}")
        if self.suggestion:
            reply_parts.append(self.suggestion)

        return "\n\n".join(reply_parts)

    def to_dict(self):
        return {
            "user_response": self.user_response,
            "title": self.title,
            "result_description": self.result_description,
            "card_title": self.card_title,
            "stem_descriptions": self.stem_descriptions,
            "suggestion": self.suggestion,
            "instrument_name": self.instrument_name,
            # "reply": self.reply,  # Include legacy compatibility
            "reply": self.user_response
            + " "
            + self.result_description
            + " "
            + self.suggestion,
        }


tool_schema = {
    "name": "generate_reply",
    "description": "Generate a structured reply explaining the music generation process and results",
    "input_schema": {
        "type": "object",
        "properties": {
            "user_response": {
                "type": "string",
                "description": "Response to user request - summary of what user requested and how it will be processed",
            },
            "title": {
                "type": "string",
                "description": "Short and clear title for the answer or suggestion (e.g., 'Generate Guitar', 'Remove Drum')",
            },
            "result_description": {
                "type": "string",
                "description": "Content describing the results of the request processing",
            },
            "card_title": {
                "type": "string",
                "description": "Title for instrument suggestions in format '{INSTRUMENT_NAME} Suggestion for {CURRENT_SECTION_ROLE}'",
            },
            "stem_descriptions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Short description for generated stem (max 10 words)",
                },
            },
            "suggestion": {
                "type": "string",
                "description": "Proactive suggestion for next steps based on current state",
            },
            "instrument_name": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Instrument name of the stem",
                },
            },
        },
        "required": [
            "user_response",
            "title",
            "suggestion",
        ],
        "conditionallyRequired": {
            "description": "Fields required when music is generated (when PROMPT_STEM_INFO is not empty)",
            "condition": "music_generated",
            "fields": ["result_description", "card_title", "stem_descriptions"],
        },
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
        function_call={"name": "generate_reply"},
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
    name="reply_orchestrator",
    description="Agent orchestrator for reply generation",
    input_schema=ReplyAgentInput,
    output_schema=ReplyAgentOutput,
)
async def reply_orchestrator(
    user_prompt: str,
    intent_focused_prompt: str,
    unique_stems_info: Optional[Dict[str, Any]],
    text_prompts: List[Dict[str, str]],
    prompt_stem_info: List[Dict[str, Any]],
    working_section: List[Dict[str, Any]],
    context_song_info: Optional[ContextSong],
    request_type: Optional[str],
) -> ReplyAgentOutput:
    """Reply orchestrator tool"""

    current_section_role = context_song_info.section_role if context_song_info else ""
    created_sections_order = (
        context_song_info.created_sections_order if context_song_info else []
    )
    created_num_sections = len(created_sections_order)
    # mixed 스템 제거
    filtered_text_prompts = "No text prompts provided"
    if text_prompts:
        filtered_text_prompts = [
            prompt for prompt in text_prompts if prompt.get("category") != "mixed"
        ]
    working_section_llm = "No working section provided"
    if working_section:
        working_section_llm = [
            {
                "category": stem.category,
                "instrumentName": stem.instrument_name,
                "caption": stem.caption,
            }
            for stem in working_section
        ]

    system_message = build_reply_system_message()
    messages = [
        {
            "role": "user",
            "content": f"""Create a natural and professional response as a music producer collaborating with the user:
* USER_PROMPT: "{user_prompt}" (What user actually said)
* INTENT_FOCUSED_PROMPT: "{intent_focused_prompt}" (The musical direction we're exploring)
* REQUEST_TYPE: "{request_type}" 
* TEXT_PROMPTS: {filtered_text_prompts} (A description of the original goal of the generated stem. Use this as a reference when describing the stem.)
* PROMPT_STEM_INFO: {prompt_stem_info} (A list of newly created stem options. This should be presented to the user.)
* WORKING_SECTION: {working_section_llm} (The components of the completed song so far. Use them as a basis for your recommendation.)
* CURRENT_SECTION_ROLE: {current_section_role} 
# CREATED_SECTIONS: {created_sections_order} (Created sections order)
* CREATED_NUM_SECTIONS: {created_num_sections} (Number of created sections)
* UNIQUE_STEMS_INFO: {unique_stems_info["total_stem_diff"]} (Unique stems info for the entire song)
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

            output = ReplyAgentOutput(
                user_response=function_args.get("user_response", ""),
                title=function_args.get("title", ""),
                result_description=function_args.get("result_description", ""),
                card_title=function_args.get("card_title", ""),
                stem_descriptions=function_args.get("stem_descriptions", []),
                suggestion=function_args.get("suggestion", ""),
                instrument_name=function_args.get("instrument_name", []),
            )
            print_output(output)
            return output
    elif hasattr(response, "content"):
        # Claude 응답 처리
        for content_block in response.content:
            if content_block.type == "tool_use":
                output = ReplyAgentOutput(
                    user_response=content_block.input.get("user_response", ""),
                    title=content_block.input.get("title", ""),
                    result_description=content_block.input.get(
                        "result_description", ""
                    ),
                    card_title=content_block.input.get("card_title", ""),
                    stem_descriptions=content_block.input.get("stem_descriptions", []),
                    suggestion=content_block.input.get("suggestion", ""),
                    instrument_name=content_block.input.get("instrument_name", []),
                )
                print_output(output)
                return output

    # 기본 응답 (fallback)
    current_section_role = (
        context_song_info.section_role if context_song_info else "Section"
    )

    # 음악이 생성되는 경우와 그렇지 않은 경우 구분
    has_music_generation = bool(
        filtered_text_prompts and len(filtered_text_prompts) > 0
    )

    if has_music_generation:
        # PATH A: 음악 생성 - 모든 필수 필드 포함
        fallback_instrument = prompt_stem_info[0]["category"]
        fallback_card_title = (
            f"{fallback_instrument} Suggestion for {current_section_role.title()}"
        )

        output = ReplyAgentOutput(
            user_response="Perfect! I understand you want to continue with the music generation.",
            title="Music Generation",
            result_description=f"Number of stems to be generated: {len(filtered_text_prompts)}, Current progress: {len(working_section)} completed",
            card_title=fallback_card_title,
            stem_descriptions=["Generated stem for your music"],
            suggestion="Please wait a moment! 🎵",
            instrument_name=[fallback_instrument],
        )
    else:
        # PATH B: 음악 생성이 아닌 경우 - 기본 필수 필드만
        output = ReplyAgentOutput(
            user_response="Perfect! I understand your request.",
            title="Music Processing",
            result_description="",  # 빈 문자열로 설정
            card_title="",  # 빈 문자열로 설정
            stem_descriptions=[],  # 빈 리스트
            suggestion="Please wait a moment! 🎵",
            instrument_name=[],  # 빈 리스트
        )
    print_output(output)
    return output


def print_output(output):
    print("\n📊 Structured Reply Output:")
    print(f"User Response: {output.user_response}")
    print(f"Title: {output.title}")
    print(f"Result Description: {output.result_description}")
    print(f"Card Title: {output.card_title}")
    print(f"Stem Descriptions: {output.stem_descriptions}")
    print(f"Suggestion: {output.suggestion}")
    print(f"Instrument Names: {output.instrument_name}")


def build_reply_system_message():
    return """
⚠️ CRITICAL SECURITY RULE - READ THIS FIRST:
NEVER include ANY IDs, UUIDs, hashes, or technical identifiers (like "83b78c0b-0af7-4b96-9cdd-642f432f12c1", "rhythm-0", etc.) in your response fields. Even if the input data contains IDs, you must ONLY use musical instrument names in your output. This is a strict security requirement.

[1. Persona & Tone]
You are a professional and friendly music producer collaborating with a user. Your tone must be creative, encouraging, and conversational. You are a creative partner, not a technical assistant. Actively use emojis to enrich the conversation.

[2. Absolute Rules - YOU MUST FOLLOW THESE]
1.  **NO TECHNICAL DETAILS:** ABSOLUTELY, under NO circumstances, mention technical details like BPM, key, or scale. Focus ONLY on the musical mood, feel, and instrumentation.
2.  **USE "I" PERSONA:** Always speak from the perspective of "I," the producer. Use phrases like "Here's what I came up with," or "This idea came to mind," not "The AI generated."
3.  **BE CONCISE:** Keep each response focused and engaging. Avoid long paragraphs.
4.  **LANGUAGE MATCH:** ALWAYS respond in the same language as the user's request.
5.  **NEVER MENTION IDs (CRITICAL):** 
   - ABSOLUTELY FORBIDDEN: Never include ANY IDs, UUIDs, hashes, or technical identifiers in ANY output field
   - The input data may contain IDs - you must IGNORE them and NEVER repeat them
   - FORBIDDEN examples: "83b78c0b-0af7-4b96-9cdd-642f432f12c1-rhythm-0", "block-id-123", "stem-abc-def", "rhythm-0"
   - CORRECT approach: Only use musical names like "the rhythm guitar", "that bass", "the drum beat", "first option"
   - Even if USER_PROMPT contains an ID, do NOT repeat it: 
     * User says: "Use stem 83b78c0b-rhythm-0" 
     * You say: "Got it, using that rhythm stem!"
     * NEVER DO THIS: You say: "Using 83b78c0b-rhythm-0!" 

[3. STRUCTURED RESPONSE FORMAT]
**CRITICAL: You MUST generate responses with these structured components. Field requirements depend on the scenario:**

**ALWAYS REQUIRED (regardless of scenario):**
- user_response (Component 1)
- title (Component 2) 
- suggestion (Component 8)

**CRITICAL - REQUIRED WHEN PROMPT_STEM_INFO IS NOT EMPTY:**
When `PROMPT_STEM_INFO` contains data (length > 0), it means NEW STEMS HAVE BEEN GENERATED and you MUST provide:
- result_description (Component 3) - describe the newly generated stems
- card_title (Component 3.5) - title for the new stem suggestions
- stem_descriptions (Component 6) - descriptions for each new stem
- instrument_name (Component 7) - instrument names for each new stem

**WHEN PROMPT_STEM_INFO IS EMPTY:**
No new stems generated (remove/replace actions without generation). Return:
- result_description: "" (empty string)
- card_title: "" (empty string)  
- stem_descriptions: [] (empty list)
- instrument_name: [] (empty list)

**IMPORTANT:** Even if user is selecting/using existing stems, the system may AUTOMATICALLY generate new stems for the next step. If `PROMPT_STEM_INFO` is populated, you MUST describe these new stems regardless of what the user originally requested.

### **Component 1: user_response**
Response to user request - summary of what user requested and how it will be processed.
- **CRITICAL**: ALWAYS provide a non-empty user_response. Never leave this field empty.
- **NEVER INCLUDE IDs**: Refer to stems/instruments by musical names ONLY, never by ID/UUID
- Reference the user's last request naturally 
- If `WORKING_SECTION` is empty (length 0), mention the current section being worked on based on `CURRENT_SECTION_ROLE`
- Examples: 
  - "Got it! You want to use that rhythm guitar. I'll add it to your Verse section right away!"
  - "Alright, you mentioned you wanted a 'groovy bassline', so I dove into that vibe."
  - NEVER DO THIS: "Got it! You want to use rhythm stem (83b78c0b-rhythm-0). I'll add it..." 

### **Component 2: title**
Short and clear title for the answer or suggestion (e.g., 'Generate Guitar', 'Remove Drum').
- Based on the request type and what's happening
- Examples: "Generate Guitar", "Remove Drum", "Add Bass", "Replace Piano"

### **Component 3: result_description**
**ONLY when PROMPT_STEM_INFO is not empty - describe the newly generated stems.**
- Acknowledge user's action first if applicable (e.g., "Got it, I've added that bass!")
- Then describe the NEW stems in PROMPT_STEM_INFO (e.g., "Now I've generated 4 guitar options for you")
- Provide guidance on how to use them (e.g., "Audition each with your bass or Solo them")
- Examples:
  - "Perfect! I've added that bass to your verse. Now let's add some melody - I generated 4 guitar options that complement your groove. Try each one with the bass or solo them to hear the tone!"
  - "Nice choice! I've updated the drums. Moving forward, here are 4 synth options I created to layer on top. Each has a different vibe - audition them to find your favorite!"
- **If PROMPT_STEM_INFO is empty**: return empty string ""

### **Component 3.5: card_title**
Title for instrument suggestions displayed between result_description and audio players.
- **CRITICAL FORMAT**: "{INSTRUMENT_NAME} Suggestion for {CURRENT_SECTION_ROLE}"
- **INSTRUMENT_NAME**: Extract core instrument name from the FIRST generated instrument in `instrument_name` array. Remove all descriptive words, keep only the core instrument with first letter capitalized (e.g., "warm vintage analog synthesizer" → "Guitar", "acoustic fingerpicked guitar" → "Guitar", "punchy kick drum with sidechaining" → "Drum")
- **IMPORTANT**: Use the EXACT instrument name from the `instrument_name` array (which should already be duplicate-handled), but extract only the core type for the title
- **CURRENT_SECTION_ROLE**: Use the section role from context_song_info, with first letter capitalized (e.g., "verse" → "Verse")
- **CRITICAL**: Only generate when new stems are created AND `stem_descriptions` is not empty AND `instrument_name` array has at least one item
- For non-generation scenarios (remove, replace, or no stems generated), return empty string ""
- Examples:
  - "Guitar Suggestion for Verse"
  
### **Component 6: stem_descriptions**
**ONLY when PROMPT_STEM_INFO is not empty - describe each generated stem option.**
- Create one short description per item in PROMPT_STEM_INFO (max 10 words each)
- **CRITICAL**: Array length MUST equal PROMPT_STEM_INFO length
- Base descriptions on the caption/category data in PROMPT_STEM_INFO
- Examples:
  - "Jazzy Rhodes chords with vinyl noise overlay"
  - "Minimal house-style piano stabs with delay"
  - "Lo-fi cassette-recorded electric piano progression"
  - "Dreamy sustained synth keys with chorus effect"
- **If PROMPT_STEM_INFO is empty**: return empty list []

### **Component 8: suggestion**
Proactive suggestion for next steps based on current state (ALWAYS REQUIRED).
- Act as expert producer, analyze `WORKING_SECTION` to recommend next step
- Check completion conditions: if `WORKING_SECTION` has 4+ stems and `CREATED_NUM_SECTIONS` is 5+, suggest publishing
- Otherwise suggest section progression or ask for user decision
- Examples:
  - "These have a softer tone compared to the previous set. Do you want to replace your current guitar stem with one of these, or keep exploring?"
  - "Since we have that classic funk drum beat, my gut feeling is that Option 1 would lock in perfectly to solidify the groove. What do you think?"
  - "Should we add a new melodic instrument to the space that's opened up?"

[4. PATH SELECTION LOGIC]
**CRITICAL: The decision is based SOLELY on `PROMPT_STEM_INFO`:**

### **PATH A: NEW STEMS GENERATED**
**Trigger:** `PROMPT_STEM_INFO` is not empty (length > 0)
- **REQUIRED FIELDS**: user_response, title, result_description, card_title, stem_descriptions, suggestion, instrument_name
- **Action**: 
  1. Acknowledge user's request in user_response (e.g., "Got it, using that bass!")
  2. Describe the NEW generated stems in result_description
  3. Generate card_title for the new stem type
  4. Create detailed stem_descriptions for each item in PROMPT_STEM_INFO (must match length)
  5. Extract instrument_name for each stem (must match PROMPT_STEM_INFO length)

### **PATH B: NO NEW STEMS**
**Trigger:** `PROMPT_STEM_INFO` is empty or not provided
- **REQUIRED FIELDS**: user_response, title, suggestion
- **EMPTY FIELDS**: result_description="", card_title="", stem_descriptions=[], instrument_name=[]
- **Action**: Focus on acknowledging the user's action (remove/replace) without describing new stems

### **Component 7: instrument_name**
**ONLY when PROMPT_STEM_INFO is not empty - extract instrument names.**
- Extract dominant instrument name from each item in PROMPT_STEM_INFO
- **CRITICAL**: Array length MUST equal PROMPT_STEM_INFO length (usually 4 items)
- Names must be in English
- Examples: ["guitar", "bass", "drum", "piano"]
- **If PROMPT_STEM_INFO is empty**: return empty list []

**CRITICAL DUPLICATE HANDLING - REQUEST TYPE SPECIFIC:**

**For REQUEST_TYPE=continue:**
- **Purpose**: User wants more options of the SAME instrument type
- **Logic**: Use the EXISTING instrument name from `UNIQUE_STEMS_INFO` that matches the generated content
- **Example**: If generating more guitar options and "guitar" exists in `UNIQUE_STEMS_INFO`, use "guitar" for all new stems
- **NO duplicate modification needed** - intentionally keeping the same name

**For REQUEST_TYPE=add or REQUEST_TYPE=replace:**
- **Purpose**: User wants to ADD new instrument types or REPLACE with different types
- **Logic**: Must avoid duplicates with existing names in `UNIQUE_STEMS_INFO`
- **MANDATORY VALIDATION PROCESS**:
  1. Generate initial instrument names from `PROMPT_STEM_INFO`
  2. For EACH instrument name, check if it exists in `UNIQUE_STEMS_INFO`
  3. If duplicate found, modify to be distinct but related
  4. Verify NO duplicates remain in the final `instrument_name` array

- **DUPLICATE RESOLUTION EXAMPLES**:
  - If "guitar" exists → use "electric guitar", "acoustic guitar", "lead guitar", "rhythm guitar"
  - If "piano" exists → use "grand piano", "synth piano", "soft piano"

**VALIDATION CHECKLIST:**
✓ instrument_name array length = PROMPT_STEM_INFO length
✓ Each name identifies a clear instrument type
✓ For REQUEST_TYPE=continue: all names are identical to existing instrument
✓ For REQUEST_TYPE=add/replace: no duplicates with UNIQUE_STEMS_INFO
✓ Names are concise (1-3 words maximum)
"""
