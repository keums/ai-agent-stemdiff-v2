
import logging
from common.cache import cache_set, cache_get, cache_delete


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    print(f"Event: {event}")
    return {
        'statusCode': 200,
        'body': "{'message': 'Disconnected'}"
    }