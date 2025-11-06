import json
import boto3
import logging
import os
from typing import Any, Dict

logger = logging.getLogger()


def send_websocket_message(connection_id: str, message: Dict[str, Any]) -> bool:
    websocket_api_endpoint = os.environ.get("WEBSOCKET_API_ENDPOINT")
    if websocket_api_endpoint:
        try:
            websocket_api_endpoint = os.environ.get("WEBSOCKET_API_ENDPOINT").replace(
                "wss://", "https://"
            )  # Assuming the endpoint is stored in the function's environment variables
            apigateway_management_api = boto3.client(
                "apigatewaymanagementapi", endpoint_url=websocket_api_endpoint
            )

            apigateway_management_api.post_to_connection(
                Data=json.dumps(message), ConnectionId=connection_id
            )
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            if "GoneException" in str(e):
                # Connection is no longer valid
                return False
            raise
    else:
        logger.debug(f"WebSocket Simulation: {message}")
        return True
