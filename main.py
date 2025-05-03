import logging
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import settings
from models import OpenAIModel, OpenAIModelList, SourceModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenAI Compatible API Proxy",
    description="Proxy server for various LLM APIs with OpenAI compatible endpoints.",
    version="0.1.0",
)

# In-memory cache for the model list
model_cache: List[SourceModel] = []
model_map: dict[str, SourceModel] = {}


async def fetch_and_cache_models():
    """Fetches the model list from the source URL and caches it."""
    global model_cache, model_map
    logger.info(f"Fetching model list from: {settings.MODEL_LIST_URL}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(settings.MODEL_LIST_URL)
            response.raise_for_status()  # Raise an exception for bad status codes
            source_models_data = response.json()

            # Validate data using Pydantic
            validated_models = [SourceModel.model_validate(m) for m in source_models_data]

            model_cache = validated_models
            model_map = {model.model_version: model for model in model_cache}
            logger.info(f"Successfully fetched and cached {len(model_cache)} models.")

        except httpx.RequestError as e:
            logger.error(f"Error fetching model list: {e}")
            # Keep stale cache if fetch fails, or handle as needed
            if not model_cache: # If cache is empty and fetch failed, raise error
                 raise HTTPException(status_code=503, detail=f"Could not fetch model list from source: {e}")
        except Exception as e:
            logger.error(f"Error processing model list: {e}")
            # Keep stale cache if processing fails
            if not model_cache:
                raise HTTPException(status_code=500, detail=f"Error processing model list: {e}")


@app.on_event("startup")
async def startup_event():
    """Fetch models on application startup."""
    await fetch_and_cache_models()
    # TODO: Add periodic refresh logic if needed


@app.get("/v1/models", response_model=OpenAIModelList)
async def get_models():
    """
    Provides a list of available models in the OpenAI API format.
    Fetches and transforms the list from the configured source URL.
    """
    if not model_cache:
        # Attempt to fetch if cache is empty (e.g., initial fetch failed)
        logger.warning("Model cache is empty, attempting to fetch again.")
        try:
            await fetch_and_cache_models()
        except HTTPException as e:
             # Return the HTTPException directly if fetch fails again
             return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
             logger.error(f"Unexpected error during fetch attempt in get_models: {e}")
             raise HTTPException(status_code=500, detail="Internal server error fetching models.")

        # Check again if cache is populated after the attempt
        if not model_cache:
             raise HTTPException(status_code=503, detail="Model list is currently unavailable.")


    openai_models = []
    for model in model_cache:
        # Determine 'owned_by' based on model_family or handle if desired
        owned_by = model.model_family # Simple example: use family name
        if "openai" in model.handle.lower():
            owned_by = "openai"
        elif "anthropic" in model.handle.lower():
            owned_by = "anthropic"
        elif "fireworks" in model.handle.lower():
             owned_by = "fireworks"
        # Add more specific rules as needed

        openai_models.append(
            OpenAIModel(
                id=model.model_version,
                owned_by=owned_by,
                # created timestamp could potentially be parsed if available in source
            )
        )

    return OpenAIModelList(data=openai_models)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

# Placeholder for the chat completions endpoint
# @app.post("/v1/chat/completions")
# async def chat_completions():
#     pass


if __name__ == "__main__":
    import uvicorn
    # For local development, run directly:
    # Set MODEL_LIST_URL env var or create .env file
    # Example: export MODEL_LIST_URL='http://127.0.0.1:8001/models.json'
    # Or create a .env file with: MODEL_LIST_URL=http://127.0.0.1:8001/models.json
    uvicorn.run(app, host="0.0.0.0", port=8000)
