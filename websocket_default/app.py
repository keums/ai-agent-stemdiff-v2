import logging
from common.cache import cache_set, cache_get, cache_delete
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    print(f"Event: {event}")

    message = json.loads(event["body"])
    dialog_uuid = message["dialogUuid"]
    cache_set(f"connection_id:{dialog_uuid}", event["requestContext"]["connectionId"])

    return {"statusCode": 200, "body": f"{'dialogUuid':{dialog_uuid}}"}
