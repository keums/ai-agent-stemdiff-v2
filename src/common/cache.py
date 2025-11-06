import json
import traceback
import boto3
import ssl
from pymemcache.client.base import Client
import os
import logging
from dotenv import load_dotenv

import requests

logger = logging.getLogger(__name__)
memcache_client = None


class MemcachedSimulatorClient:
    url: str

    def __init__(self, url: str):
        self.url = url

    def get(self, key: str):
        response = requests.get(f"{self.url}/get/{key}")
        text = json.loads(response.text)
        return text.encode("utf-8")

    def set(self, key: str, value: bytes, expire: int):
        response = requests.post(
            f"{self.url}/set",
            json={"key": key, "value": value.decode("utf-8"), "expire": expire},
        )
        return response.json()


def get_memcache_client():
    global memcache_client
    if memcache_client is None:
        try:
            MEMCACHED_ENDPOINT = os.environ.get("MEMCACHED_ENDPOINT")
            if MEMCACHED_ENDPOINT:
                context = ssl.create_default_context()
                host, port = MEMCACHED_ENDPOINT.split(":")
                memcache_client = Client((host, int(port)), tls_context=context)
            else:
                print("MEMCACHED_ENDPOINT is not set. Using memcache simulator.")
                memcache_client = MemcachedSimulatorClient(
                    os.environ.get("MEMCACHED_SIMULATOR_URL")
                )
        except Exception as e:
            print(f"Error getting memcache client: {e}")
            print(traceback.format_exc())
    return memcache_client


def cache_set(key, value, ttl=60 * 60 * 24):
    client = get_memcache_client()
    if client is None:
        logger.warning(
            f"Memcache client not available, skipping cache_set for key: {key}"
        )
        return False

    try:
        if isinstance(value, dict) or isinstance(value, list):
            value = json.dumps(value)
        # Ensure value is properly encoded as UTF-8 bytes
        if isinstance(value, str):
            value = value.encode("utf-8")
        client.set(key, value, expire=ttl)
        return True
    except Exception as e:
        logger.error(f"Error setting cache for key {key}: {e}")
        return False


def cache_get(key):
    client = get_memcache_client()
    if client is None:
        logger.warning(
            f"Memcache client not available, skipping cache_get for key: {key}"
        )
        return None

    try:
        value = client.get(key)
        if value is not None:
            return value.decode("utf-8")
        return None
    except Exception as e:
        logger.error(f"Error getting cache for key {key}: {e}")
        logger.error(traceback.format_exc())
        return None
