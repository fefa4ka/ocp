import logging
from typing import List, Any, Dict, AsyncGenerator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from models import (
    OpenAIModel, OpenAIModelList, SourceModel, SourceModelList,
    OpenAIChatCompletionRequest # Added import
)

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

    headers = {}
    if settings.MODEL_LIST_AUTH_TOKEN:
        headers["Authorization"] = f"OAuth {settings.MODEL_LIST_AUTH_TOKEN}"
        logger.info("Using OAuth token for model list request.")
    else:
        logger.info("No auth token found for model list request.")


    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(settings.MODEL_LIST_URL, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            source_data = response.json()

            # Validate data using Pydantic, expecting {"models": [...]}
            validated_source_list = SourceModelList.model_validate(source_data)

            model_cache = validated_source_list.models
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxies chat completion requests to the backend defined by the model's handle,
    using the same host as the MODEL_LIST_URL and the same OAuth token.
    """
    try:
        request_data = await request.json()
        # Optional: Validate request body structure against OpenAIChatCompletionRequest
        # try:
        #     OpenAIChatCompletionRequest.model_validate(request_data)
        # except Exception as e:
        #     logger.warning(f"Request body validation failed: {e}")
        #     raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        model_id = request_data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        logger.info(f"Received chat completion request for model: {model_id}")

        # Find the model in the cached map
        target_model = model_map.get(model_id)
        if not target_model:
            logger.warning(f"Model '{model_id}' not found in cache. Available: {list(model_map.keys())}")
            # Optionally, refresh cache and retry?
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

        handle = target_model.handle
        logger.info(f"Found model '{model_id}' with handle: {handle}")

        # --- Determine the target backend URL ---
        try:
            parsed_list_url = urlparse(settings.MODEL_LIST_URL)
            if not parsed_list_url.scheme or not parsed_list_url.netloc:
                raise ValueError("MODEL_LIST_URL is invalid or missing scheme/host")

            # Construct target URL using the scheme and host from MODEL_LIST_URL and the handle
            target_url = f"{parsed_list_url.scheme}://{parsed_list_url.netloc}{handle}"
            logger.info(f"Constructed target URL: {target_url}")

        except ValueError as e:
             logger.error(f"Could not parse MODEL_LIST_URL ('{settings.MODEL_LIST_URL}'): {e}")
             raise HTTPException(status_code=500, detail="Server configuration error: Invalid MODEL_LIST_URL.")


        # --- Prepare Headers ---
        headers_to_forward = {
            "Content-Type": "application/json",
            # Pass through other relevant headers if needed, be careful about sensitive ones
            # "Accept": request.headers.get("Accept", "application/json"),
        }

        # Add Authorization header using the same token as for the model list
        if settings.MODEL_LIST_AUTH_TOKEN:
            headers_to_forward["Authorization"] = f"OAuth {settings.MODEL_LIST_AUTH_TOKEN}"
            logger.debug("Using configured MODEL_LIST_AUTH_TOKEN for backend request.")
        else:
            # If no token is configured, should we proceed? Depends on backend requirements.
            # For this specific case (same host/auth), lack of token might imply an issue.
            logger.warning(f"No MODEL_LIST_AUTH_TOKEN configured. Request to {target_url} will be unauthenticated.")
            # raise HTTPException(status_code=500, detail="Server configuration error: Missing auth token for backend.")


        # --- Forward the request ---
        is_streaming = request_data.get("stream", False)
        logger.info(f"Forwarding request for model '{model_id}' to {target_url}. Streaming: {is_streaming}")

        # Define the stream generator here, accepting necessary parameters
        async def stream_generator(req_url: str, req_data: dict, req_headers: dict) -> AsyncGenerator[bytes, None]:
            # Create the client *inside* the generator to manage its lifecycle correctly
            async with httpx.AsyncClient() as stream_client:
                try:
                    async with stream_client.stream(
                        "POST",
                        req_url,
                        json=req_data,
                        headers=req_headers,
                        timeout=180.0
                    ) as backend_response:
                        # Check for backend errors *before* streaming body
                        if backend_response.status_code >= 400:
                            error_body = await backend_response.aread()
                            error_detail = error_body.decode() or f"Backend error {backend_response.status_code}"
                            logger.error(f"Backend streaming request failed with status {backend_response.status_code}: {error_detail}")
                            # Raising here will be caught by Starlette's error handling for streaming responses
                            raise HTTPException(
                                status_code=backend_response.status_code,
                                detail=error_detail
                            )

                        # Stream the response body chunk by chunk
                        async for chunk in backend_response.aiter_bytes():
                            yield chunk
                    logger.info(f"Finished streaming response from backend for model '{model_id}'")
                except Exception as e:
                    # Log errors occurring during the streaming process itself
                    logger.error(f"Error during backend stream processing for model {model_id}: {e}")
                    # Re-raise to signal an internal server error during streaming
                    # This will likely terminate the stream prematurely for the client.
                    raise

        # Now, handle the request based on streaming flag
        try:
            if is_streaming:
                # Create the generator instance, passing the required arguments
                generator = stream_generator(target_url, request_data, headers_to_forward)
                # Return a StreamingResponse using the generator
                # OpenAI standard content type for streaming is text/event-stream
                return StreamingResponse(generator, media_type="text/event-stream")

            else: # Non-streaming request
                # Use a separate client instance for non-streaming requests
                async with httpx.AsyncClient() as client:
                    backend_response = await client.post(
                        target_url,
                        json=request_data, # Forward the original request body
                        headers=headers_to_forward,
                        timeout=180.0 # Set a reasonable timeout for LLM requests
                    )

                    # Raise exception for 4xx/5xx responses from the backend
                    backend_response.raise_for_status()

                    # Return the raw JSON response from the backend
                    response_data = backend_response.json()
                    logger.info(f"Successfully received non-streaming response from backend for model '{model_id}'")
                    # Return backend's status code and content directly
                    return JSONResponse(content=response_data, status_code=backend_response.status_code)

            except httpx.RequestError as e:
                logger.error(f"Error requesting backend {target_url}: {e}")
                raise HTTPException(status_code=503, detail=f"Error contacting backend service: {e}")
            except httpx.HTTPStatusError as e: # Caught for non-streaming errors
                 logger.error(f"Backend service at {target_url} returned error {e.response.status_code}: {e.response.text}")
                 try:
                     error_content = e.response.json()
                 except Exception:
                     error_content = {"detail": e.response.text} # Fallback if response is not JSON
                 return JSONResponse(content=error_content, status_code=e.response.status_code)

    except HTTPException:
         raise # Re-raise FastAPI HTTP exceptions (including those from stream_generator)
    except Exception as e:
        logger.exception(f"Unexpected error in chat_completions endpoint: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    # For local development, run directly:
    # Set MODEL_LIST_URL env var or create .env file
    # Example: export MODEL_LIST_URL='http://127.0.0.1:8001/models.json'
    # Or create a .env file with: MODEL_LIST_URL=http://127.0.0.1:8001/models.json
    uvicorn.run(app, host="0.0.0.0", port=8000)
