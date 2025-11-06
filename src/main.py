# main.py
"""
AI Agent StemDiff - Main Orchestration Module

This module serves as the central orchestrator for the AI Agent StemDiff system, which
processes user prompts to generate and manipulate music stems through intelligent
conversation and memory management.

Key Components:
- Music Agent: Analyzes user intent and generates music information
- Reply Agent: Generates conversational responses
- Memory Management: Maintains context across conversation turns
- Stem Search: Finds relevant music stems from Elasticsearch
- Stem Generation: Creates new stem variations using AI models

Main Functions:
- main(): Primary orchestration function that processes user prompts
- extract_data_from_memory(): Retrieves and processes conversation context

The system supports both local development and production deployment with
different logging levels and data storage mechanisms.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime

import aiohttp
import boto3
import requests
import sentry_sdk
from dotenv import load_dotenv
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration

import tools.embedding.music_text_joint_embedding as music_text_joint_embedding
import tools.embedding.text_embedding as text_embedding
import tools.search.stem_search as search_stems_from_es
from agents.composition_agent import composition_agent
from agents.intent_agent import intent_agent_router
from agents.music_agent import generate_music_info
from agents.reply_agent import reply_orchestrator
from agents.reply_for_publish import reply_for_publish_song
from common.cache import cache_get, cache_set
from common.websocket import send_websocket_message
from memory.memory_loader import (
    extract_data_from_memory,
    get_recent_valid_dialog_memory,
)
from models import ContextSong, GenerateStemDiffOutput, Stem
from tools.search.search_thumbnail import search_thumbnail
from tools.search.stem_search import SearchStemOutput
from utils.get_stem_section_info import update_mix_stems

CANDIDATE_MEMORY_COUNT = 15
# Load environment variables
logger = logging.getLogger(__name__)
is_local = os.getenv("WEBSOCKET_API_ENDPOINT") is None
# is_local = False
# Configure logging with handler - set root to INFO to reduce noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration(), AwsLambdaIntegration()],
    traces_sample_rate=1.0,
)

# Enable DEBUG for our internal modules only
if is_local:
    # Load environment variables if run in local
    load_dotenv()
    # Enable DEBUG for main module only - suppress operational logs
    logger.setLevel(logging.DEBUG)
    # Suppress operational INFO logs from internal modules
    logging.getLogger("utils.data_schema_mapper").setLevel(logging.WARNING)
    logging.getLogger("tools.mcp_base").setLevel(logging.WARNING)

else:
    logger.setLevel(logging.INFO)

# External libraries will use INFO level by default (set above)
# Only our main module will show DEBUG logs when is_local=True


OUTPUT_FORMAT = "aac"

DIALOG_API_URL = os.getenv("DIALOG_API_URL")
USER_API_URL = os.getenv("USER_API_URL")


def lambda_handler(event, context):
    """
    Handle payload data from the event and call the main function
    """
    print(f"is_local: {is_local}")

    #
    try:
        stage = "handle_payload_data"
        logging.debug("Event: %s", event)
        body = event.get("body", "")
        logging.info("Body: %s", body)
        if isinstance(body, str):
            body = json.loads(body)

        dialog_uuid = body.get("dialogUuid")
        if not dialog_uuid:
            raise Exception("Dialog UUID is required")
        session_uuid = cache_get(f"session_uuid:{dialog_uuid}")
        if not session_uuid:
            raise Exception("Session UUID is required")
        request_uuid = cache_get(f"request_uuid:{dialog_uuid}")
        if not request_uuid:
            raise Exception("Request UUID is required")

        remix_song_info = cache_get(f"remix_song_info:{dialog_uuid}")
        if remix_song_info:
            remix_song_info = json.loads(remix_song_info)
        user_id = cache_get(f"user_id:{dialog_uuid}")
        chat_message = cache_get(f"chat_message:{dialog_uuid}")
        media = cache_get(f"media:{dialog_uuid}")

        dialogs_in_session = cache_get(f"dialogs_in_session:{session_uuid}")
        if dialogs_in_session:
            dialogs_in_session = json.loads(dialogs_in_session)
        else:
            dialogs_in_session = []
        dialogs_in_session.append(dialog_uuid)
        cache_set(f"dialogs_in_session:{session_uuid}", dialogs_in_session)

        sentry_sdk.set_context("dialog_uuid", dialog_uuid)
        sentry_sdk.set_context("session_uuid", session_uuid)
        sentry_sdk.set_context("request_uuid", request_uuid)
        sentry_sdk.set_context("dialogs_in_session", dialogs_in_session)
        sentry_sdk.set_context("user_id", user_id)
        sentry_sdk.set_context("chat_message", chat_message)
        sentry_sdk.set_context("dialogs_in_session", dialogs_in_session)

        logging.debug("Dialog UUID: %s", dialog_uuid)
        logging.debug("Session UUID: %s", session_uuid)
        logging.debug("Request UUID: %s", request_uuid)
        logging.debug("Dialogs in session: %s", dialogs_in_session)

    except Exception as e:
        logging.error("Error handling payload data: %s", e)
        return handle_error(e, stage, dialog_uuid, request_uuid)

    try:
        result = asyncio.run(
            main(
                chat_message,
                request_uuid,
                dialog_uuid,
                session_uuid,
                user_id,
                remix_song_info,
            )
        )
        if result.get("statusCode", 200) == 500:
            return result
    except Exception as e:
        # error handling is done in main function
        # shouldn't reach here
        return {"statusCode": 500, "body": "Internal server error"}

    return {
        "statusCode": 200,
        "body": json.dumps({"result": "success"}),
    }


async def main(
    user_prompt,
    request_uuid,
    dialog_uuid,
    session_uuid,
    user_id,
    remix_song_info=None,
    send_backend_message=True,
):
    # TODO: make a class for remix_song_info
    # remix_song_info = {"remix_song_id": "m000328_wegoup_1", "remix_section_name": "C"}

    error_message = ""
    logger.debug("\n\n\n ===== 🚀🚀🚀🚀 Dialog_id: %s =====", dialog_uuid)

    if not session_uuid:
        logging.error("Session UUID is required")
        return handle_error(
            Exception("Session UUID is required"),
            "session_uuid_required",
            dialog_uuid,
            request_uuid,
        )

    usage = []
    start_time = time.time()
    # TODO CHECK LIST FOR NEXT STEP
    # 1. Memory
    # 2. Context info.
    # 3. Block Prompts
    # 4. selected_song_id
    # 5. Block Prompt Agent
    # 6. selected_block_prompt
    # 6. Memory Agent
    # 7. Prompt Stem Search
    # 8. Context Song Search
    # 9. Update Stems
    # 10. Reply for Gen.

    # * 🔍🔍 Read Recent memory
    memory_data = get_recent_valid_dialog_memory(
        dialog_uuid, session_uuid, CANDIDATE_MEMORY_COUNT
    )

    # * 🔍🔍 Intent Agent(Router)
    # Example intent_result
    # intent_result = IntentResult(
    #     request_type="chat",
    #     intent_focused_prompt="User is asking what the chatbot can do",
    #     response="I can help you create music...",
    # )
    intent_result = await intent_agent_router(user_prompt, memory_data)

    print(f"User request type: {intent_result.request_type}")
    print(f"Intent focused prompt: {intent_result.intent_focused_prompt}")

    # If it's a chat request, return the response immediately
    if intent_result.request_type == "chat" and intent_result.response:
        print(f"Chat response: {intent_result.response}")
        #! NEED TO BE CHECKED
        # TODO: Return chat response to user through websocket or appropriate channel
        request_answer = format_answer(
            request_uuid,
            memory_data.turn_index + 1,
            session_uuid,
            dialog_uuid,
            user_prompt,
            error_message=error_message,
            working_section_index=memory_data.working_section_index,
            intent_focused_prompt=intent_result.intent_focused_prompt,
            intent_history=memory_data.intent_history.append(
                intent_result.intent_focused_prompt
            ),
            chosen_sections=memory_data.chosen_sections,
            context_song_info=memory_data.context_song,
            reply=intent_result.response,
        )

        cache_set(f"request_answer:{dialog_uuid}", request_answer)

        # send websocket result
        data = {"status": "completed", "logs": [], "data": request_answer}
        connection_id = cache_get(f"connection_id:{request_uuid}")
        if connection_id is not None:
            send_websocket_message(connection_id, data)
            logger.info("Sent websocket result")

        if send_backend_message:
            try:
                response = save_answer(request_answer)
                print(response)
            except Exception as e:
                logger.info("Error saving answer: %s", e)
                return handle_error(e, "save_answer", dialog_uuid, request_uuid)

        return request_answer

    if intent_result.request_type == "generate":
        composition_result = await composition_agent(
            intent_result.intent_focused_prompt, memory_data
        )

    return
    # TODO: 3. Contextual Agent
    # TODO: 4. Context Prompt Agent
    # TODO: 5. Block Prompt Agent
    # TODO: 6. Memory Agent
    # TODO: 7. Prompt Stem Search
    # TODO: 8. Context Song Search
    # TODO: 9. Update Stems
    # TODO: 10. Reply for Gen.
    # TODO: 10. Reply for Publish Song
    # TODO: 10. Reply for Chat

    # Get related data from the memory
    try:
        memory_data, strategy, llm_usage, previous_working_section_index = (
            await extract_data_from_memory(
                user_prompt, dialog_uuid, session_uuid, is_local
            )
        )
        usage.append({"function": "supervisor_agent", "callLLM": llm_usage})
    except Exception as e:
        logger.info("Error in extract_data_from_memory function %s", e)
        return handle_error(e, "extract_data_from_memory", dialog_uuid, request_uuid)
    end_time = time.time()
    print(f"TIME: Extract_data_from_memory: {end_time - start_time} seconds")

    # %% #? 🔍🔍  Generate music info

    if strategy.is_publish_song:
        # TODO:: Mixing and returning URL from the mixer server
        audio_url = ""
        # TODO:: audio_url -> music_caption
        music_caption = ""

        try:
            reply_orchestrator_result = await reply_for_publish_song(
                user_prompt=user_prompt,
                intent_focused_prompt=strategy.intent_focused_prompt,
                mix_stem_diff=memory_data.chosen_sections,
                context_song_info=memory_data.context_song,
                caption_input="",
            )
            usage.append({"function": "reply_orchestrator", "callLLM": 1})
            thumbnail_url = await search_thumbnail(
                f"{reply_orchestrator_result.reply}"
                + f"* Genre: {reply_orchestrator_result.genre}"
                + f"* Mood: {reply_orchestrator_result.mood}"
                + f"* Instrument: {reply_orchestrator_result.instruments}"
            )
            reply_orchestrator_result.thumbnail_url = thumbnail_url.thumbnail_url

            # TODO

            request_answer = format_answer(
                request_uuid,
                memory_data.turn_index,
                session_uuid,
                dialog_uuid,
                user_prompt,
                error_message=error_message,
                working_section_index=memory_data.working_section_index,
                intent_focused_prompt=strategy.intent_focused_prompt,
                intent_history=memory_data.intent_history,
                text_prompts=None,
                target_music_information=None,
                # mix_stem_diff=generate_music_info_result.selected_stem_diff,
                chosen_sections=memory_data.chosen_sections,
                context_song_info=memory_data.context_song,
                prompt_stem_info=None,
                output_uris=[],
                reply=reply_orchestrator_result.to_dict()["reply"],
                thumbnail_url=reply_orchestrator_result.thumbnail_url,
                title=reply_orchestrator_result.title,
                status="published",
            )

            cache_set(f"request_answer:{dialog_uuid}", request_answer)

            # send websocket result
            data = {"status": "completed", "logs": [], "data": request_answer}
            connection_id = cache_get(f"connection_id:{request_uuid}")
            if connection_id is not None:
                send_websocket_message(connection_id, data)
                logger.info("Sent websocket result")

            if send_backend_message:
                try:
                    response = save_answer(request_answer)
                    print(response)
                except Exception as e:
                    logger.info("Error saving answer: %s", e)
                    return handle_error(e, "save_answer", dialog_uuid, request_uuid)

                try:
                    response = calculate_usage(
                        user_id,
                        session_uuid,
                        dialog_uuid,
                        request_uuid,
                        usage,
                        [],
                    )
                    print(response)
                except Exception as e:
                    logger.info("Error calculating usage: %s", e)
                    return handle_error(
                        e, "calculate_usage_error", dialog_uuid, request_uuid
                    )

            return request_answer

        except Exception as e:
            error_message = f"❌ Error processing reply orchestrator: {str(e)}"
            print(error_message)
            return handle_error(e, "reply_orchestrator", dialog_uuid, request_uuid)
    start_time = time.time()
    try:
        generate_music_info_result, llm_usage = await generate_music_info(
            user_prompt=user_prompt,
            intent_focused_prompt=strategy.intent_focused_prompt,
            prev_context_song_info=memory_data.context_song,
            prev_generated_stems=memory_data.generated_stems,
            chosen_sections=memory_data.chosen_sections,
            intent_history=memory_data.intent_history,
            working_section=memory_data.working_section,
        )
        target_context_section_index = None
        if strategy.is_start_new_section:
            if generate_music_info_result.selected_stem_diff:
                memory_data.chosen_sections = update_mix_stems(
                    chosen_sections=memory_data.chosen_sections,
                    selected_stem_diff=generate_music_info_result.selected_stem_diff,
                    # working_section_index=memory_data.working_section_index,
                    is_new_section=strategy.is_start_new_section,
                    working_section_index=previous_working_section_index,
                    request_type=generate_music_info_result.request_type,
                )

            memory_data.chosen_sections = update_mix_stems(
                chosen_sections=memory_data.chosen_sections,
                selected_stem_diff={},
                is_new_section=strategy.is_start_new_section,
                working_section_index=memory_data.working_section_index,
                request_type=generate_music_info_result.request_type,
            )

            memory_data.working_section = memory_data.chosen_sections[
                memory_data.working_section_index
            ]
        else:
            if (
                strategy.target_working_section_index is not None
                and strategy.target_working_section_index
                != memory_data.working_section_index
            ):
                memory_data.working_section_index = (
                    strategy.target_working_section_index
                )
                target_context_section_index = strategy.target_working_section_index

            if generate_music_info_result.selected_stem_diff:
                memory_data.chosen_sections = update_mix_stems(
                    chosen_sections=memory_data.chosen_sections,
                    selected_stem_diff=generate_music_info_result.selected_stem_diff,
                    is_new_section=strategy.is_start_new_section,
                    working_section_index=memory_data.working_section_index,
                    request_type=generate_music_info_result.request_type,
                )

                memory_data.working_section = memory_data.chosen_sections[
                    memory_data.working_section_index
                ]

        connection_id = cache_get(f"connection_id:{request_uuid}")
        if connection_id is not None:
            send_websocket_message(
                connection_id, {"systemMessage": "Music info generation completed"}
            )
        usage.append({"function": "generate_mix_info", "callLLM": llm_usage})
        updated_generate_music_info_result = generate_music_info_result.to_dict()
        updated_generate_music_info_result["context_song_info"] = (
            memory_data.context_song.to_dict() if memory_data.context_song else None
        )
        updated_generate_music_info_result["mix_stem_diff"] = (
            memory_data.chosen_sections
        )
        updated_generate_music_info_result["working_section_index"] = (
            memory_data.working_section_index
        )
    except Exception as e:
        logger.info("Error processing music info: %s", e)
        return handle_error(e, "generate_music_info", dialog_uuid, request_uuid)
    end_time = time.time()
    print(f"TIME: Generate_music_info: {end_time - start_time} seconds")
    start_time = time.time()
    if generate_music_info_result.request_type not in ["remove", "replace"]:
        try:
            text_embedding_result = text_embedding.get_text_embedding(
                {"text_prompts": generate_music_info_result.text_prompts}
            )
            usage.append({"function": "get_text_embedding", "callLLM": 1})
            music_text_joint_embedding_result = (
                await music_text_joint_embedding.get_music_text_joint_embedding(
                    {"text_prompts": generate_music_info_result.text_prompts}
                )
            )
            usage.append({"function": "get_music_text_joint_embedding", "callLLM": 1})
        except Exception as e:
            logger.info("Error processing text embedding: %s", e)
            return handle_error(e, "get_text_embedding", dialog_uuid, request_uuid)
        end_time = time.time()
        print(f"TIME: Getting embedding: {end_time - start_time} seconds")
        start_time = time.time()
        try:
            result_search: search_stems_from_es.SearchStemOutput = (
                await search_stems_from_es.search_stems_from_es(
                    text_embeddings=text_embedding_result.text_embedding,
                    music_text_joint_embeddings=music_text_joint_embedding_result.music_text_joint_embedding,
                    context_song_info=memory_data.context_song,
                    target_music_info=generate_music_info_result.target_music_info,
                    request_type=generate_music_info_result.request_type,
                    continue_stem_info=generate_music_info_result.continue_stem_info,
                    remix_song_info=remix_song_info,
                    dialog_uuid=dialog_uuid,
                    target_context_section_index=target_context_section_index,
                )
            )

            connection_id = cache_get(f"connection_id:{request_uuid}")
            if connection_id is not None:
                send_websocket_message(
                    connection_id, {"systemMessage": "Stem search completed"}
                )
            if (
                getattr(memory_data.context_song, "is_remix", False)
                or result_search.vocal_stem_info
            ):
                if strategy.is_start_new_section or strategy.is_start_from_scratch:
                    memory_data.chosen_sections = update_mix_stems(
                        chosen_sections=memory_data.chosen_sections,
                        selected_stem_diff=result_search.vocal_stem_info,
                        is_new_section=strategy.is_start_new_section,
                        working_section_index=memory_data.working_section_index,
                        request_type=generate_music_info_result.request_type,
                    )

        except Exception as e:
            logger.info("Error processing search stems from es: %s", e)
            return handle_error(e, "search_stems_from_es", dialog_uuid, request_uuid)
    else:
        # remove or replace
        result_search = SearchStemOutput(
            prompt_stem_info=[],
            context_song_info=memory_data.context_song,
            vocal_stem_info=[],
        )

    updated_result_search = result_search.to_dict()
    updated_result_search["mix_stem_diff"] = memory_data.chosen_sections

    # Update with search_stems_from_es result
    # %% #? reply agent
    #! NEED TO BE PARALLELIZED (reply_orchestrator and stem_diff)
    try:
        reply_orchestrator_result = await reply_orchestrator(
            user_prompt=user_prompt,
            request_type=generate_music_info_result.request_type,
            unique_stems_info=generate_music_info_result.unique_stems_info,
            intent_focused_prompt=strategy.intent_focused_prompt,
            text_prompts=generate_music_info_result.text_prompts,
            prompt_stem_info=result_search.prompt_stem_info,
            working_section=memory_data.working_section,
            context_song_info=result_search.context_song_info,
        )
        usage.append({"function": "reply_orchestrator", "callLLM": 1})

    except Exception as e:
        error_message = f"❌ Error processing reply orchestrator: {str(e)}"
        print(error_message)
        raise Exception(error_message)

    end_time = time.time()
    print(f"TIME: Reply_orchestrator: {end_time - start_time} seconds")
    start_time = time.time()
    # %%  #? 🔍🔍 stem diff
    generate_stem_diff_result = None
    if generate_music_info_result.request_type not in ["remove", "replace"]:
        #! NEED TO BE PARALLELIZED (reply_orchestrator and stem_diff)
        try:
            if result_search.context_song_info is None:
                print("⚠️ Context song info is None, skipping stem diff generation")
            else:
                generate_stem_diff_result: GenerateStemDiffOutput = (
                    await generate_stem_diff(
                        context_song=result_search.context_song_info,
                        prompt_stem_info=result_search.prompt_stem_info,
                        mix_stem_diff_list=memory_data.working_section,
                        task_id=dialog_uuid,
                    )
                )
                context_song_info = result_search.context_song_info
                duration = (
                    context_song_info.bar_count * 4 * (60 / context_song_info.bpm)
                )
                usage.append(
                    {
                        "function": "generate_stemdiff",
                        "billedDuration": len(generate_stem_diff_result.output_uris)
                        * duration,
                    }
                )
                connection_id = cache_get(f"connection_id:{request_uuid}")
                if connection_id is not None:
                    send_websocket_message(
                        connection_id, {"systemMessage": "Stem generation completed"}
                    )
        except Exception as e:
            error_message = f"❌ Error processing generate stem diff: {str(e)}"
            print(error_message)

    # todo: prevent empty sections
    # non_empty_sections = [section for section in memory_data.chosen_sections if section]
    # memory_data.chosen_sections = non_empty_sections
    # Save the final schema
    end_time = time.time()
    print(f"TIME: Stemdiff: {end_time - start_time} seconds")
    start_time = time.time()

    request_answer = format_answer(
        request_uuid,
        memory_data.turn_index,
        session_uuid,
        dialog_uuid,
        user_prompt,
        error_message=error_message,
        working_section_index=memory_data.working_section_index,
        intent_focused_prompt=strategy.intent_focused_prompt,
        intent_history=memory_data.intent_history,
        text_prompts=generate_music_info_result.text_prompts,
        target_music_information=generate_music_info_result.target_music_info,
        # mix_stem_diff=generate_music_info_result.selected_stem_diff,
        chosen_sections=memory_data.chosen_sections,
        context_song_info=result_search.context_song_info,
        prompt_stem_info=result_search.prompt_stem_info,
        output_uris=(
            generate_stem_diff_result.output_uris if generate_stem_diff_result else []
        ),
        reply=reply_orchestrator_result.to_dict()["reply"],
    )

    cache_set(f"request_answer:{dialog_uuid}", request_answer)

    # send websocket result
    data = {"status": "completed", "logs": [], "data": request_answer}
    connection_id = cache_get(f"connection_id:{request_uuid}")
    if connection_id is not None:
        send_websocket_message(connection_id, data)
        logger.info("Sent websocket result")

    if send_backend_message:
        try:
            response = save_answer(request_answer)
            print(response)
        except Exception as e:
            logger.info("Error saving answer: %s", e)
            return handle_error(e, "save_answer", dialog_uuid, request_uuid)

        try:
            used_stems = get_used_stems(result_search)
            response = calculate_usage(
                user_id, session_uuid, dialog_uuid, request_uuid, usage, used_stems
            )
            print(response)
        except Exception as e:
            logger.info("Error calculating usage: %s", e)
            return handle_error(e, "calculate_usage_error", dialog_uuid, request_uuid)

    return request_answer


def get_used_stems(result_search):
    prompt_stems = [stem["id"] for stem in result_search.prompt_stem_info]
    context_stems = [
        uri.split("/")[-1] for uri in result_search.context_song_info.context_audio_uris
    ]
    return prompt_stems + context_stems


async def generate_stem_diff(
    context_song: ContextSong,
    prompt_stem_info: list[dict],
    mix_stem_diff_list: list[Stem],
    task_id: str,
):
    context_song_info = context_song.to_dict()
    context_song_info["context_audio_uris"] = [
        uri.split(".")[0] for uri in context_song_info["context_audio_uris"]
    ]
    mix_stem_diff = [stem.to_dict() for stem in mix_stem_diff_list]
    for stem in mix_stem_diff:
        stem["uri"] = stem["uri"].split(".")[0]
    for stem in prompt_stem_info:
        stem["uri"] = stem["uri"].split(".")[0]
    async with aiohttp.ClientSession() as session:
        async with session.post(
            os.getenv("STEM_DIFF_REQEUST_HELPER_API_URL", "") + "/request_stem_diff",
            json={
                "context_song_info": context_song_info,
                "prompt_stem_info": prompt_stem_info,
                "mix_stem_diff": mix_stem_diff,
                "task_id": task_id,
                "env": os.getenv("ENVIRONMENT"),
            },
        ) as response:
            result = await response.json()
            print("🔍 check result 🔍")
            print(result)
            return GenerateStemDiffOutput(
                output_uris=[
                    ((uri + "." + OUTPUT_FORMAT) if (uri[-3:] != "aac") else uri)
                    for uri in result["output_uris"]
                ]
            )


def handle_error(error: Exception, stage, dialog_uuid, request_uuid):
    logger.info("Error: %s", error)
    logger.info("Stage: %s", stage)
    logger.info("Dialog UUID: %s", dialog_uuid)
    connection_id = cache_get(f"connection_id:{request_uuid}")
    logger.info("Connection ID: %s", connection_id)
    if connection_id is not None:
        try:
            event_id = sentry_sdk.capture_exception(error)
            logger.info("Event_id: %s", event_id)
            send_websocket_message(
                connection_id,
                {
                    "status": "error",
                    "stage": f"{stage}",
                    "event_id": event_id,
                    "dialog_uuid": dialog_uuid,
                },
            )
        except Exception as e:
            logger.info("Error sending websocket message: %s", e)
            raise Exception(f"Error sending websocket message: {e}")
    else:
        logger.info("handle_error:No connection id found")

    response = requests.post(
        f"{os.getenv('ERROR_API_URL')}",
        json={
            "dialogUuid": dialog_uuid,
            "eventId": event_id,
        },
    )
    if response.status_code != 200:
        raise Exception(f"Error sending error: {response}")
    else:
        logger.info("Error sent to error api")

    return {
        "statusCode": 500,
        "body": json.dumps(
            {"error": f"{error}", "event_id": event_id, "dialog_uuid": dialog_uuid}
        ),
    }


def save_answer(answer):
    print("SAVING Answer: ", answer)
    response = requests.post(
        f"{DIALOG_API_URL}/dialog/answer",
        json=answer,
        headers={
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if response.status_code not in [201, 200]:
        raise Exception(f"Error saving answer: {response}")
    return response.text


def calculate_usage(user_id, session_uuid, dialog_uuid, request_uuid, usage, stems):
    data = {
        "userId": user_id,
        "sessionUuid": session_uuid,
        "dialogUuid": dialog_uuid,
        "requestUuid": request_uuid,
        "stems": stems,
        "tools": usage,
    }

    print("USER_API_URL: ", USER_API_URL)
    response = requests.post(
        f"{USER_API_URL}/token/usage/save",
        json=data,
        timeout=30,
    )
    if response.status_code not in [201, 200]:
        raise Exception(f"Error calculating usage: {response}")
    return response.text


def check_minimum_token(user_id):
    response = requests.get(
        f"{USER_API_URL}/token/usage/check/{user_id}",
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"Error checking minimum token: {response}")
    return response.text


def format_answer(
    request_uuid,
    request_turn_index,
    session_uuid,
    dialog_uuid,
    user_prompt,
    error_message="",
    working_section_index=0,
    intent_focused_prompt="",
    intent_history=None,
    text_prompts=None,
    target_music_information=None,
    chosen_sections=None,  # list of list of Stem
    context_song_info: ContextSong | None = None,
    prompt_stem_info=None,  # list of dict
    output_uris=None,
    reply={},
    thumbnail_url=None,
    title=None,
    status="completed",
):
    print("request_uuid:", request_uuid)
    print("request_turn_index:", request_turn_index)
    print("session_uuid:", session_uuid)
    print("dialog_uuid:", dialog_uuid)
    print("user_prompt:", user_prompt)
    print("error_message:", error_message)
    print("working_section_index:", working_section_index)
    print("intent_focused_prompt:", intent_focused_prompt)
    print("intent_history:", intent_history)
    print("text_prompts:", text_prompts)
    print("target_music_information:", target_music_information)
    print("chosen_sections:", chosen_sections)
    print("context_song_info:", context_song_info)
    print("prompt_stem_info:", prompt_stem_info)
    print("output_uris:", output_uris)
    print("reply:", reply)
    print("thumbnail_url:", thumbnail_url)
    print("title:", title)
    print("status:", status)
    generated_mix_id = str(uuid.uuid4())

    def upsert(obj, key, value):
        obj[key] = value
        # obj.__setattr__(key, value)
        return obj

    def id_from_stem_uri(uris):
        return [uri.split("/")[-1].split(".")[0] for uri in uris]

    bar_count = 0
    for section in chosen_sections:
        bar_count += max([0] + [stem.bar_count for stem in section])
    bpm = context_song_info.bpm if context_song_info else 0
    if bpm == 0:
        if len(chosen_sections) > 0:
            if len(chosen_sections[0]) > 0:
                bpm = chosen_sections[0][0].bpm
    if bpm == 0:
        bpm = 90
        logger.info("BPM is 0, using default 90")
    seconds_per_beat = 60 / bpm
    duration = bar_count * 4 * seconds_per_beat
    status_message = status
    if error_message:
        status_message = "error"
    elif status == "published":
        status_message = "completed"

    request = {
        "uuid": request_uuid,
        "sessionUuid": session_uuid,
        "taskId": request_uuid,
        "speaker": "user",
        "dataType": "mix",
        "turnIndex": request_turn_index,  # 1 if it's the first from the session
        "thumb": False,
        "status": status_message,
        "tokenUsageId": [],
        "requestUuid": None,
        "processingTimeMs": 0,
        "schemaVersion": "1.2",
        "errorMessage": "",
        "createdAt": datetime.now().isoformat(),
        "medias": [],
        "chatMessage": {
            "uuid": str(uuid.uuid4()),
            "dialogUuid": request_uuid,
            "createdAt": datetime.now().isoformat(),
            "chatText": user_prompt,
            "intentFocusedPrompt": intent_focused_prompt,
        },
        # "context": {
        #     "previousContext": intent_history,
        # },
        "intentHistory": intent_history,
        "requestInformation": {
            "globalMusicInformation": target_music_information,
            "stemPrompts": text_prompts,
            "contextSongInfo": (
                context_song_info.to_camel_case_dict() if context_song_info else None
            ),
            "workingSectionIndex": working_section_index,
        },
        "answers": [
            {
                "uuid": dialog_uuid,
                "sessionUuid": session_uuid,
                "taskId": request_uuid,
                "speaker": "assistant",
                "dataType": "mix",
                "chatMessage": {
                    "uuid": str(uuid.uuid4()),
                    "dialogUuid": dialog_uuid,
                    "createdAt": datetime.now().isoformat(),
                    "chatText": (
                        reply
                        # reply.get("reply", "")
                        # json.dumps(reply, ensure_ascii=False)
                        # if isinstance(reply, dict)
                        # else str(reply)
                    ),
                },
                "createdAt": datetime.now().isoformat(),
                "turnIndex": request_turn_index + 1,
                "requestUuid": request_uuid,
                "thumb": False,
                "status": status_message,
                "tokenUsageId": [],
                "processingTimeMs": 0,
                "schemaVersion": "1.2",
                "errorMessage": error_message,
                "mix": (
                    {
                        "mixData": {
                            "id": generated_mix_id,
                            "dialogUuid": dialog_uuid,
                            "status": "error" if error_message else status,
                            "duration": duration,
                            "bpm": context_song_info.bpm if context_song_info else 120,
                            "key": context_song_info.key if context_song_info else "",
                            "stems": [
                                [
                                    upsert(stem.to_dict(), "mixId", generated_mix_id)
                                    for stem in section
                                ]
                                for section in chosen_sections
                            ],
                            "thumbnail": thumbnail_url or None,
                            "title": title or None,
                        }
                    }
                    if chosen_sections
                    else {"mixData": {"stems": [[]]}}
                ),
                "suggestedStems": (
                    [
                        {
                            "id": stem["id"],
                            "mixId": None,
                            "dialogUuid": dialog_uuid,
                            "isOriginal": False,
                            "isBlock": True,
                            "category": stem["stemType"],
                            "caption": stem["caption"],
                            "sectionName": (
                                context_song_info.section_name
                                if context_song_info
                                else ""
                            ),
                            "sectionRole": (
                                context_song_info.section_role
                                if context_song_info
                                else ""
                            ),
                            "barCount": (
                                context_song_info.bar_count if context_song_info else 0
                            ),
                            "bpm": context_song_info.bpm if context_song_info else 120,
                            "key": context_song_info.key if context_song_info else "",
                            "uri": output_uris[i] if output_uris else None,
                            "url": (
                                get_presigned_url(output_uris[i])
                                if output_uris
                                else None
                            ),
                            "usedBlockIds": id_from_stem_uri(
                                context_song_info.context_audio_uris
                                if context_song_info
                                else []
                            ),
                        }
                        for i, stem in enumerate(prompt_stem_info)
                    ]
                    if prompt_stem_info
                    else []
                ),
            }
        ],
    }
    # print("check request:", request)
    # suggested_stems_count = len(request["answers"][0]["suggestedStems"])

    # if len(reply.get("instrument_name", [])) != suggested_stems_count:
    #     print("⚠️ Warning: Instrument name of stem length mismatch.")
    #     # instrument_name_of_stem의 길이가 suggestedStems보다 짧을 때, 같은 값을 복사
    #     instrument_names = reply.get("instrument_name", [])

    #     # suggestedStems의 길이만큼 instrument_name을 복사
    #     if instrument_names:
    #         # 마지막 값을 복사하여 길이를 맞춤
    #         last_instrument = instrument_names[-1]
    #         while len(instrument_names) < suggested_stems_count:
    #             instrument_names.append(last_instrument)

    #     # 복사된 instrument_names로 업데이트
    #     for idx, stem in enumerate(request["answers"][0]["suggestedStems"]):
    #         if idx < len(instrument_names):
    #             request["answers"][0]["suggestedStems"][idx]["instrumentName"] = (
    #                 instrument_names[idx]
    #             )
    # else:
    #     for idx, stem in enumerate(request["answers"][0]["suggestedStems"]):
    #         request["answers"][0]["suggestedStems"][idx]["instrumentName"] = reply.get(
    #             "instrument_name"
    #         )[idx]
    return request


def get_presigned_url(uri):
    return boto3.client("s3").generate_presigned_url(
        "get_object",
        Params={
            "Bucket": uri.split("/")[2],
            "Key": "/".join(uri.split("/")[3:]),
        },
    )


async def test_main():
    session_uuid = "2016"
    dialog_uuid = str(uuid.uuid4())
    dialogs_in_session = cache_get(f"dialogs_in_session:{session_uuid}")
    print("dialogs_in_session: ", dialogs_in_session)
    if dialogs_in_session:
        dialogs_in_session = json.loads(dialogs_in_session)
    else:
        dialogs_in_session = []
    dialogs_in_session.append(dialog_uuid)
    cache_set(f"dialogs_in_session:{session_uuid}", dialogs_in_session)

    remix_song_info = {
        "remix_song_id": "u001917_let_it_go",
        # "remix_song_id": "u000008_money",
        "remix_section_name": "A",
    }

    try:
        result = await main(
            # "Use the first rhythm",
            "Create a dance song",
            # "aa658054-4f08-4050-9ea7-42cb75a3c4fc",
            # str(uuid.uuid4()),
            # "bc28d03a-3c35-4003-9c55-ed7522d8d895",
            str(uuid.uuid4()),
            dialog_uuid,
            session_uuid,
            "44",
            remix_song_info,
            False,
        )
        print("Test result:", result)
    except Exception as e:
        logger.info("Error in test_main: %s", e)


if __name__ == "__main__":
    # For testing purposes only
    asyncio.run(test_main())
