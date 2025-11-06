# FastAPI Application

import json
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.cache import cache_get, cache_set

# Import main processing function
from main import main


# Pydantic models for API
class TextPrompt(BaseModel):
    category: str
    text: str
    uri: str = ""


class TextPrompts(BaseModel):
    text_prompts: List[TextPrompt]


class ContextSongInfo(BaseModel):
    song_id: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    bar_count: Optional[int] = None
    section_name: Optional[str] = None
    section_role: Optional[str] = None
    structure_info: Optional[Dict[str, str]] = {}
    context_audio_uris: Optional[List[str]] = []
    generated_mix_uris: Optional[List[str]] = []


class StemDiffRequest(BaseModel):
    message: str
    context_song_info: Dict[str, Any]
    generated_stem_diff_uris: List[str]
    mix_stem_diff: List[Dict[str, Any]]


class StemDiffResponse(BaseModel):
    output_uris: List[str]
    prompt_stem_info: List[Dict[str, Any]]
    context_song_info: Optional[Dict[str, Any]] = None
    mix_stem_diff: List[List[Dict[str, Any]]]
    message: str
    working_section_index: Optional[int] = None


# FastAPI app
app = FastAPI(
    title="AI Agent Stem Diff API",
    description="AI Agent for generating stem differences based on text prompts",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",  # Vue default port
        "http://127.0.0.1:8080",
        "*",  # Allow all origins in development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_stem_diff(
    message: str,
    request_uuid: str,
    user_id: str,
    dialog_uuid: str,
    session_uuid: str,
    remix_song_info: Dict[str, Any],
) -> StemDiffResponse:
    try:
        dialogs_in_session = cache_get(f"dialogs_in_session:{session_uuid}")
        if dialogs_in_session:
            dialogs_in_session = json.loads(dialogs_in_session)
        else:
            dialogs_in_session = []
        dialogs_in_session.append(dialog_uuid)
        cache_set(f"dialogs_in_session:{session_uuid}", dialogs_in_session)

        # Get result from memcache which already has presigned URLs
        result = await main(
            message,
            request_uuid,
            dialog_uuid,
            session_uuid,
            user_id,
            remix_song_info,
            send_backend_message=False,
        )

        if not result.get("answers", []):
            # No answers case - return early
            print(result.get("mix_stem_diff", []))
            return StemDiffResponse(
                output_uris=[],
                prompt_stem_info=[],
                context_song_info=None,
                mix_stem_diff=result.get("mix_stem_diff", [[]]),
                message=result.get("message", ""),
                working_section_index=None,
            )

        # Extract data from memcache result - URLs are already included
        suggested_stems = result["answers"][0]["suggestedStems"]
        mix_stem_diff = None
        if result["answers"][0].get("mix") and result["answers"][0]["mix"].get(
            "mixData", {}
        ).get("stems"):
            mix_stem_diff = result["answers"][0]["mix"]["mixData"]["stems"]

        # Extract authenticated URLs directly from suggestedStems (already has 'url' field)
        authenticated_urls = [stem.get("url", "") for stem in suggested_stems]

        print("\n🔗 Authenticated URLs from memcache: \n", authenticated_urls)

        return StemDiffResponse(
            output_uris=authenticated_urls,
            prompt_stem_info=suggested_stems,
            context_song_info=result["requestInformation"]["contextSongInfo"],
            mix_stem_diff=(mix_stem_diff if mix_stem_diff is not None else [[]]),
            message=result["answers"][0]["chatMessage"]["chatText"],
            working_section_index=result["requestInformation"]["workingSectionIndex"],
        )

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail=f"Error processing stem diff: {str(e)}"
        )


@app.get("/")
async def root():
    return {"message": "AI Agent Stem Diff API is running! 🚀"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is ready to process requests"}


@app.post("/generate-stem-diff", response_model=StemDiffResponse)
async def generate_stem_diff(request: StemDiffRequest):
    """
    Generate stem differences based on text prompts and context song information.

    - **text_prompts**: List of text prompts with category, text, and URI
    - **context_song_info**: Context information about the song including audio URIs
    """

    dialog_uuid = str(uuid.uuid4())
    request_uuid = str(uuid.uuid4())
    user_id = "1234567890"
    # session_uuid = str(uuid.uuid4())
    #! FOR TEST ONLY
    session_uuid = "e8c9d585-0730-4017-a0b2-69a8c73ab3g1"
    remix_song_info = {
        # "remix_song_id": "u001917_let_it_go",
        "remix_song_id": "u000028_trouble",
        # "remix_song_id": "u000012_money",
        # "remix_song_id": "u000033_get_to_me",
        # "remix_section_name": "C",
    }  #! only 1st step of remix process
    remix_song_info = None
    return await process_stem_diff(
        request.message,
        request_uuid,
        user_id,
        dialog_uuid,
        session_uuid,
        remix_song_info,
    )


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
