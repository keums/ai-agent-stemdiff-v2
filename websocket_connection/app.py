import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    print(f"Event: {event}")
    return {"statusCode": 200, "body": "{'message': 'Connected'}"}
