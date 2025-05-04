#!/usr/bin/env python3
"""
Test script for Cohere API integration.
This script sends a direct request to the Cohere API to understand the correct format.
"""

import os
import json
import httpx
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get auth token from environment
auth_token = os.environ.get("MODEL_LIST_AUTH_TOKEN")
if not auth_token:
    logger.error("MODEL_LIST_AUTH_TOKEN environment variable not set")
    exit(1)

# Cohere API endpoint
cohere_url = "https://api.eliza.yandex.net/cohere/v2/chat"

# Test different request formats
test_formats = [
    {
        "name": "Format 1: messages array",
        "payload": {
            "messages": [
                {"role": "USER", "content": "Hello, how are you?"}
            ],
            "model": "command-r"
        }
    },
    {
        "name": "Format 2: message object",
        "payload": {
            "message": {"role": "USER", "content": "Hello, how are you?"},
            "model": "command-r"
        }
    },
    {
        "name": "Format 3: chat_history",
        "payload": {
            "message": "Hello, how are you?",
            "chat_history": [],
            "model": "command-r"
        }
    }
]

async def test_cohere_api():
    """Test different request formats with the Cohere API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"OAuth {auth_token}"
    }
    
    async with httpx.AsyncClient() as client:
        for test in test_formats:
            logger.info(f"Testing {test['name']}")
            logger.debug(f"Request payload: {test['payload']}")
            
            try:
                response = await client.post(
                    cohere_url,
                    json=test['payload'],
                    headers=headers,
                    timeout=30.0
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    logger.info(f"Success! Response structure: {json.dumps(response_data, indent=2)}")
                else:
                    logger.error(f"Error response: {response.text}")
                    
            except Exception as e:
                logger.exception(f"Error testing format: {e}")
            
            logger.info("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cohere_api())
