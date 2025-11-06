import os
import json
import logging
import traceback
import uuid

import boto3
from src.common.cache import cache_get, cache_set


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

cors_headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
}


def lambda_handler(event, context):
    print(f"MEMCACHED_ENDPOINT: {os.environ.get('MEMCACHED_ENDPOINT')}")
    try:
        print(f"Event: {event}")
        # return CORS headers for preflight requests
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 200, "headers": cors_headers, "body": "OK"}
        body = event.get("body", "")
        # api gateway body is base64 encoded
        # decode base64 if needed
        if event.get("isBase64Encoded"):
            import base64

            body = base64.b64decode(body).decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        print(f"Body: {body}")
        if not body:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": "Body is required",
            }

        request_uuid = body.get("requestUuid")
        session_uuid = body.get("sessionUuid")
        chat_message = body.get("chatMessage")
        remix_song_info = body.get("remixSongInfo")
        user_id = body.get("userId")
        debug_mode = body.get("debug", "0")
        dialog_uuid = str(uuid.uuid4())

        if not request_uuid or not session_uuid or not chat_message or not user_id:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": "requestUuid, sessionUuid, chatMessage, and userId are required",
            }

        # store request data in cache
        cache_set(f"session_uuid:{dialog_uuid}", session_uuid)
        cache_set(f"debug_mode:{dialog_uuid}", debug_mode)
        cache_set(f"chat_message:{dialog_uuid}", chat_message)
        cache_set(f"request_uuid:{dialog_uuid}", request_uuid)
        cache_set(f"user_id:{dialog_uuid}", user_id)
        if remix_song_info:
            cache_set(f"remix_song_info:{dialog_uuid}", remix_song_info)

        dialogs_in_session = cache_get(f"dialogs_in_session:{session_uuid}")
        if dialogs_in_session:
            dialogs_in_session = json.loads(dialogs_in_session)
        if not dialogs_in_session:
            dialogs_in_session = []
        if dialog_uuid not in dialogs_in_session:
            print(f"Dialog UUID: {dialog_uuid} not in dialogs_in_session")
            dialogs_in_session.append(dialog_uuid)
            cache_set(f"dialogs_in_session:{session_uuid}", dialogs_in_session)

        media = body.get("media", [])
        if media:
            cache_set(f"media:{dialog_uuid}", media)

        lambda_client = boto3.client("lambda")
        lambda_client.invoke(
            FunctionName=os.environ.get("MAIN_HANDLER_FUNCTION_NAME"),
            InvocationType="Event",
            Payload=json.dumps({"body": {"dialogUuid": dialog_uuid}}),
        )

        WEBSOCKET_API_ENDPOINT = os.environ.get("WEBSOCKET_API_ENDPOINT")
        return {
            "statusCode": 200,
            "headers": cors_headers,
            "body": json.dumps(
                {
                    "result": "success",
                    "dialogUuid": dialog_uuid,
                    "requestUuid": request_uuid,
                    "sessionUuid": session_uuid,
                    "chatMessage": chat_message,
                    "media": media,
                    "websocket": f"{WEBSOCKET_API_ENDPOINT}",
                    "debug": debug_mode,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error in agent_front: {e}")
        logger.error(traceback.format_exc())
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": "Internal server error",
        }
