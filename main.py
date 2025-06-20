import base64
import json  # Add json import
import logging
import time  # Add time import
from typing import Any, AsyncGenerator, Dict, List, Optional  # Import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from models import (
    OpenAIChatCompletionRequest,  # Added import
    OpenAIImageData,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
    OpenAIModel,
    OpenAIModelList,
    SourceModel,
    SourceModelList,
)

# --- Logging Setup ---
# Map string level names to logging constants
log_level_str = settings.LOG_LEVEL.upper()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NONE": logging.CRITICAL + 1, # Effectively disable logging
}
log_level = log_level_map.get(log_level_str, logging.INFO) # Default to INFO if invalid

# Configure basic logging
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Force=True might be needed if uvicorn also configures logging
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
# --- End Logging Setup ---


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

    # Validate URL is configured
    if not settings.MODEL_LIST_URL:
        logger.error("MODEL_LIST_URL is not configured")
        raise HTTPException(status_code=500, detail="MODEL_LIST_URL is not configured")

    headers = {}
    if settings.MODEL_LIST_AUTH_TOKEN:
        headers["Authorization"] = f"OAuth {settings.MODEL_LIST_AUTH_TOKEN}"
        logger.info("Using OAuth token for model list request.")
    else:
        logger.warning("No auth token found for model list request.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Making request to {settings.MODEL_LIST_URL} with headers: {list(headers.keys())}")
            response = await client.get(settings.MODEL_LIST_URL, headers=headers)
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()  # Raise an exception for bad status codes
            source_data = response.json()
            logger.debug(f"Raw response data keys: {list(source_data.keys()) if isinstance(source_data, dict) else 'Not a dict'}")

            # Validate data using Pydantic, expecting {"models": [...]}
            validated_source_list = SourceModelList.model_validate(source_data)
            logger.info(f"Validated {len(validated_source_list.models)} models from source")

            # Filter out only models with handle "unavailable"
            filtered_models = [
                model for model in validated_source_list.models
                if model.handle != "unavailable"
            ]

            # Log how many models were filtered out
            filtered_count = len(validated_source_list.models) - len(filtered_models)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} unavailable models.")

            model_cache = filtered_models
            model_map = {model.model_version: model for model in model_cache}
            logger.info(f"Successfully fetched and cached {len(model_cache)} models.")
            
            # Log first few model IDs for verification
            if model_cache:
                sample_models = [m.model_version for m in model_cache[:5]]
                logger.info(f"Sample model IDs: {sample_models}")

        except httpx.RequestError as e:
            logger.error(f"Network error fetching model list: {e}")
            # Keep stale cache if fetch fails, or handle as needed
            if not model_cache: # If cache is empty and fetch failed, raise error
                 raise HTTPException(status_code=503, detail=f"Could not fetch model list from source: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching model list: {e.response.status_code} - {e.response.text}")
            if not model_cache:
                raise HTTPException(status_code=503, detail=f"Model list service returned {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error processing model list: {e}")
            logger.exception("Full traceback:")
            # Keep stale cache if processing fails
            if not model_cache:
                raise HTTPException(status_code=500, detail=f"Error processing model list: {e}")


@app.on_event("startup")
async def startup_event():
    """Fetch models on application startup."""
    await fetch_and_cache_models()
    # TODO: Add periodic refresh logic if needed


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """
    Proxies embedding requests to the backend defined by the model's handle,
    using the same host as the MODEL_LIST_URL and the same OAuth token.
    """
    try:
        # 1. Read the original request data
        original_request_data = await request.json()
        logger.debug(f"Received raw request data for embeddings: {original_request_data}")

        # 2. Make a copy to modify for the backend request payload
        payload_for_backend = original_request_data.copy()

        # 3. Perform model lookup using the model from the payload
        model_id = payload_for_backend.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        logger.info(f"Received embeddings request for model: {model_id}")

        # Find the model in the cached map
        target_model = model_map.get(model_id)
        if not target_model:
            logger.warning(f"Model '{model_id}' not found in cache. Available: {list(model_map.keys())}")
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
            logger.info(f"Constructed target URL for embeddings: {target_url}")

        except ValueError as e:
            logger.error(f"Could not parse MODEL_LIST_URL ('{settings.MODEL_LIST_URL}'): {e}")
            raise HTTPException(status_code=500, detail="Server configuration error: Invalid MODEL_LIST_URL.")

        # --- Prepare Headers ---
        headers_to_forward = {
            "Content-Type": "application/json",
        }

        # Add Authorization header using the same token as for the model list
        if settings.MODEL_LIST_AUTH_TOKEN:
            headers_to_forward["Authorization"] = f"OAuth {settings.MODEL_LIST_AUTH_TOKEN}"
            logger.debug("Using configured MODEL_LIST_AUTH_TOKEN for backend request.")
        else:
            logger.warning(f"No MODEL_LIST_AUTH_TOKEN configured. Request to {target_url} will be unauthenticated.")

        # --- Forward the request ---
        logger.info(f"Forwarding embeddings request for model '{model_id}' to {target_url}")

        async with httpx.AsyncClient() as client:
            backend_response = await client.post(
                target_url,
                json=payload_for_backend,
                headers=headers_to_forward,
                timeout=180.0  # Set a reasonable timeout for embedding requests
            )

            # Raise exception for 4xx/5xx responses from the backend
            backend_response.raise_for_status()

            # Return the raw JSON response from the backend
            response_data = backend_response.json()
            logger.info(f"Successfully received response from backend for model '{model_id}'")
            logger.debug(f"Backend response data for model '{model_id}': {response_data}")
            logger.debug(f"Raw request body: {payload_for_backend}")
            logger.debug(f"Raw response body: {backend_response.text}")

            # --- Transform response if needed ---
            final_response_data = response_data
            # Check if this is an OpenAI response with a nested 'response' field
            if "/openai/" in handle.lower() and "response" in response_data:
                logger.info(f"Extracting nested response for OpenAI model '{model_id}'")
                final_response_data = response_data.get("response", {})

            return JSONResponse(content=final_response_data, status_code=backend_response.status_code)

    except httpx.RequestError as e:
        logger.error(f"Error requesting backend: {e}")
        error_response = {
            "error": {
                "message": f"Error contacting backend service: {e}",
                "type": "server_error",
                "param": None,
                "code": "service_unavailable"
            }
        }
        return JSONResponse(content=error_response, status_code=503)
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend service returned error {e.response.status_code}: {e.response.text}")
        try:
            error_content = e.response.json()
            if "error" in error_content and isinstance(error_content["error"], dict):
                return JSONResponse(content=error_content, status_code=e.response.status_code)
            else:
                error_message = error_content.get("detail", error_content.get("message", str(error_content)))
                if isinstance(error_message, dict):
                    error_message = json.dumps(error_message)
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                        "param": None,
                        "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                    }
                }
                return JSONResponse(content=error_response, status_code=e.response.status_code)
        except Exception:
            error_response = {
                "error": {
                    "message": e.response.text or f"Backend error {e.response.status_code}",
                    "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                    "param": None,
                    "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                }
            }
            return JSONResponse(content=error_response, status_code=e.response.status_code)
    except HTTPException as e:
        logger.error(f"HTTP exception in embeddings endpoint: {e.detail} (status: {e.status_code})")
        error_response = {
            "error": {
                "message": str(e.detail),
                "type": "server_error" if e.status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": "service_unavailable" if e.status_code >= 500 else "bad_request"
            }
        }
        return JSONResponse(content=error_response, status_code=e.status_code)
    except Exception as e:
        logger.exception(f"Unexpected error in embeddings endpoint: {e}")
        error_response = {
            "error": {
                "message": "An unexpected error occurred",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }
        return JSONResponse(content=error_response, status_code=500)


@app.options("/v1/models")
async def options_models():
    """Handle OPTIONS request for CORS preflight and return model list."""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400"
    }
    
    # Return the same model list as GET method
    logger.info(f"OPTIONS /v1/models called. Current cache size: {len(model_cache)}")
    
    if not model_cache:
        # Attempt to fetch if cache is empty (e.g., initial fetch failed)
        logger.warning("Model cache is empty, attempting to fetch again.")
        try:
            await fetch_and_cache_models()
        except HTTPException as e:
             logger.error(f"HTTPException during model fetch: {e.status_code} - {e.detail}")
             return JSONResponse(status_code=e.status_code, content={"detail": e.detail}, headers=headers)
        except Exception as e:
             logger.error(f"Unexpected error during fetch attempt in options_models: {e}")
             error_response = {
                 "error": {
                     "message": "Internal server error fetching models.",
                     "type": "server_error",
                     "param": None,
                     "code": "internal_error"
                 }
             }
             return JSONResponse(content=error_response, status_code=500, headers=headers)

        # Check again if cache is populated after the attempt
        if not model_cache:
             logger.error("Model cache is still empty after fetch attempt")
             error_response = {
                 "error": {
                     "message": "Model list is currently unavailable.",
                     "type": "server_error",
                     "param": None,
                     "code": "service_unavailable"
                 }
             }
             return JSONResponse(content=error_response, status_code=503, headers=headers)

    openai_models = []
    for model in model_cache:
        # Skip image generation models in the /v1/models endpoint response
        # but keep them in the cache for image generation requests
        if model.model_family.lower() in ["dall-e-3", "recraft", "ideogram"]:
            continue
                
        # Include embedding models in the models list
        # They will be shown in the models list but handled by the embeddings endpoint
            
        # Determine 'owned_by' based on model_family or handle if desired
        owned_by = model.model_family # Simple example: use family name
        if "openai" in model.handle.lower():
            owned_by = "openai"
        elif "anthropic" in model.handle.lower():
            owned_by = "anthropic"
        elif "fireworks" in model.handle.lower():
             owned_by = "fireworks"
        elif "gemini" in model.handle.lower():
             owned_by = "google"
        # Add more specific rules as needed

        openai_models.append(
            OpenAIModel(
                id=model.model_version,
                object="model",
                created=int(time.time()),
                owned_by=owned_by,
            )
        )

    model_list = OpenAIModelList(data=openai_models)
    return JSONResponse(content=model_list.model_dump(), status_code=200, headers=headers)


@app.get("/v1/models", response_model=OpenAIModelList)
async def get_models():
    """
    Provides a list of available models in the OpenAI API format.
    Fetches and transforms the list from the configured source URL.
    """
    logger.info(f"GET /v1/models called. Current cache size: {len(model_cache)}")
    logger.info(f"MODEL_LIST_URL: {settings.MODEL_LIST_URL}")
    logger.info(f"MODEL_LIST_AUTH_TOKEN configured: {'Yes' if settings.MODEL_LIST_AUTH_TOKEN else 'No'}")
    
    if not model_cache:
        # Attempt to fetch if cache is empty (e.g., initial fetch failed)
        logger.warning("Model cache is empty, attempting to fetch again.")
        try:
            await fetch_and_cache_models()
        except HTTPException as e:
             logger.error(f"HTTPException during model fetch: {e.status_code} - {e.detail}")
             return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e:
             logger.error(f"Unexpected error during fetch attempt in get_models: {e}")
             raise HTTPException(status_code=500, detail="Internal server error fetching models.")

        # Check again if cache is populated after the attempt
        if not model_cache:
             logger.error("Model cache is still empty after fetch attempt")
             raise HTTPException(status_code=503, detail="Model list is currently unavailable.")


    openai_models = []
    for model in model_cache:
        # Skip image generation models in the /v1/models endpoint response
        # but keep them in the cache for image generation requests
        if model.model_family.lower() in ["dall-e-3", "recraft", "ideogram"]:
            continue
                
        # Include embedding models in the models list
        # They will be shown in the models list but handled by the embeddings endpoint
            
        # Determine 'owned_by' based on model_family or handle if desired
        owned_by = model.model_family # Simple example: use family name
        if "openai" in model.handle.lower():
            owned_by = "openai"
        elif "anthropic" in model.handle.lower():
            owned_by = "anthropic"
        elif "fireworks" in model.handle.lower():
             owned_by = "fireworks"
        elif "gemini" in model.handle.lower():
             owned_by = "google"
        # Add more specific rules as needed

        openai_models.append(
            OpenAIModel(
                id=model.model_version,
                object="model",
                created=int(time.time()),
                owned_by=owned_by,
            )
        )

    return OpenAIModelList(data=openai_models)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """
    Proxies image generation requests to the backend defined by the model's handle,
    using the same host as the MODEL_LIST_URL and the same OAuth token.
    Supports models from ideogram, dall-e-3, and recraft families.
    """
    try:
        # 1. Read the original request data
        original_request_data = await request.json()
        logger.debug(f"Received raw request data for image generation: {original_request_data}")

        # 2. Validate request against our model
        try:
            validated_request = OpenAIImageGenerationRequest.model_validate(original_request_data)
            logger.debug(f"Validated image generation request: {validated_request}")
        except Exception as e:
            logger.warning(f"Image generation request validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        # 3. Make a copy to modify for the backend request payload
        payload_for_backend = original_request_data.copy()

        # 4. Perform model lookup using the model from the payload
        model_id = payload_for_backend.get("model", "dall-e-3")  # Default to dall-e-3 if not specified
        logger.info(f"Received image generation request for model: {model_id}")

        # Find the model in the cached map
        target_model = model_map.get(model_id)
        if not target_model:
            logger.warning(f"Model '{model_id}' not found in cache. Available: {list(model_map.keys())}")
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

        handle = target_model.handle
        model_family = target_model.model_family.lower()
        logger.info(f"Found model '{model_id}' with handle: {handle}, family: {model_family}")

        # Check if model family is supported for image generation
        if model_family not in ["ideogram", "dall-e-3", "recraft"]:
            logger.error(f"Model family '{model_family}' is not supported for image generation")
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' with family '{model_family}' is not supported for image generation"
            )

        # Log detailed model information
        logger.debug(f"Model details - ID: {model_id}, Family: {model_family}, Handle: {handle}")
        logger.debug(f"Full model object: {target_model.model_dump()}")

        # --- Determine the target backend URL ---
        try:
            parsed_list_url = urlparse(settings.MODEL_LIST_URL)
            if not parsed_list_url.scheme or not parsed_list_url.netloc:
                raise ValueError("MODEL_LIST_URL is invalid or missing scheme/host")

            # Construct target URL using the scheme and host from MODEL_LIST_URL and the handle
            target_url = f"{parsed_list_url.scheme}://{parsed_list_url.netloc}{handle}"
            logger.info(f"Constructed target URL for image generation: {target_url}")

        except ValueError as e:
            logger.error(f"Could not parse MODEL_LIST_URL ('{settings.MODEL_LIST_URL}'): {e}")
            raise HTTPException(status_code=500, detail="Server configuration error: Invalid MODEL_LIST_URL.")

        # --- Prepare Headers ---
        headers_to_forward = {
            "Content-Type": "application/json",
        }

        # Add Authorization header using the same token as for the model list
        if settings.MODEL_LIST_AUTH_TOKEN:
            headers_to_forward["Authorization"] = f"OAuth {settings.MODEL_LIST_AUTH_TOKEN}"
            logger.debug("Using configured MODEL_LIST_AUTH_TOKEN for backend request.")
        else:
            logger.warning(f"No MODEL_LIST_AUTH_TOKEN configured. Request to {target_url} will be unauthenticated.")

        # --- API Specific Adjustments ---
        # Transform the payload based on the target backend
        if model_family == "ideogram":
            payload_for_backend = transform_openai_request_to_ideogram(payload_for_backend)
        elif model_family == "recraft":
            payload_for_backend = transform_openai_request_to_recraft(payload_for_backend)
        elif model_family == "dall-e-3":
            # For dall-e-3, we need to ensure the model parameter is correct
            # The error shows the backend expects 'dall-e-3', 'dall-e-2', or 'gpt-image-1'
            original_model = payload_for_backend.get("model", "")
            logger.debug(f"Original model parameter: {original_model}")

            # Extract the base model name without parameters
            if ":" in original_model:
                base_model = original_model.split(":")[0]
                logger.debug(f"Extracted base model: {base_model}")
                payload_for_backend["model"] = base_model

            logger.debug(f"Adjusted model parameter for dall-e-3 family: {payload_for_backend.get('model')}")

        logger.debug(f"Transformed payload for {model_family}: {payload_for_backend}")

        # --- Forward the request ---
        logger.info(f"Forwarding image generation request for model '{model_id}' to {target_url}")

        async with httpx.AsyncClient() as client:
            backend_response = await client.post(
                target_url,
                json=payload_for_backend,
                headers=headers_to_forward,
                timeout=180.0  # Set a reasonable timeout for image generation
            )

            # Raise exception for 4xx/5xx responses from the backend
            backend_response.raise_for_status()

            # Return the raw JSON response from the backend
            response_data = backend_response.json()
            logger.info(f"Successfully received response from backend for model '{model_id}'")
            logger.debug(f"Backend response data for model '{model_id}': {response_data}")

            # Log additional details about the response
            if "data" in response_data:
                logger.info(f"Generated {len(response_data['data'])} images")
                for i, img_data in enumerate(response_data["data"]):
                    if "url" in img_data:
                        logger.info(f"Image {i+1} URL available: {img_data['url'][:30]}...")
                    elif "b64_json" in img_data:
                        b64_length = len(img_data["b64_json"]) if img_data["b64_json"] else 0
                        logger.info(f"Image {i+1} base64 data available: {b64_length} bytes")

            # --- Transform response if needed ---
            final_response_data = response_data.get('response', {})
            if model_family == "ideogram":
                final_response_data = transform_ideogram_response_to_openai(response_data, model_id)
            elif model_family == "recraft":
                final_response_data = transform_recraft_response_to_openai(response_data, model_id)
            # dall-e-3 uses OpenAI format, so no transformation needed

            # Validate the response against our model
            try:
                OpenAIImageGenerationResponse.model_validate(final_response_data)
            except Exception as e:
                logger.error(f"Response validation failed: {e}")
                # Continue anyway, just log the error

            return JSONResponse(content=final_response_data, status_code=backend_response.status_code)

    except httpx.RequestError as e:
        logger.error(f"Error requesting backend: {e}")
        error_response = {
            "error": {
                "message": f"Error contacting backend service: {e}",
                "type": "server_error",
                "param": None,
                "code": "service_unavailable"
            }
        }
        return JSONResponse(content=error_response, status_code=503)
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend service returned error {e.response.status_code}: {e.response.text}")

        # Try to parse and log the error response in more detail
        try:
            error_json = e.response.json()
            logger.error(f"Parsed error response: {error_json}")

            # Extract and log specific error details if available
            if isinstance(error_json, dict):
                if "error" in error_json:
                    error_details = error_json["error"]
                    logger.error(f"Error type: {error_details.get('type')}")
                    logger.error(f"Error message: {error_details.get('message')}")
                    logger.error(f"Error param: {error_details.get('param')}")
                    logger.error(f"Error code: {error_details.get('code')}")
                elif "response" in error_json and "error" in error_json.get("response", {}):
                    error_details = error_json["response"]["error"]
                    logger.error(f"Error type: {error_details.get('type')}")
                    logger.error(f"Error message: {error_details.get('message')}")
                    logger.error(f"Error param: {error_details.get('param')}")
                    logger.error(f"Error code: {error_details.get('code')}")
        except Exception as parse_error:
            logger.error(f"Failed to parse error response: {parse_error}")
        try:
            error_content = e.response.json()
            if "error" in error_content and isinstance(error_content["error"], dict):
                return JSONResponse(content=error_content, status_code=e.response.status_code)
            else:
                error_message = error_content.get("detail", error_content.get("message", str(error_content)))
                if isinstance(error_message, dict):
                    error_message = json.dumps(error_message)
                error_response = {
                    "error": {
                        "message": error_message,
                        "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                        "param": None,
                        "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                    }
                }
                return JSONResponse(content=error_response, status_code=e.response.status_code)
        except Exception:
            error_response = {
                "error": {
                    "message": e.response.text or f"Backend error {e.response.status_code}",
                    "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                    "param": None,
                    "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                }
            }
            return JSONResponse(content=error_response, status_code=e.response.status_code)
    except HTTPException as e:
        logger.error(f"HTTP exception in images_generations: {e.detail} (status: {e.status_code})")
        error_response = {
            "error": {
                "message": str(e.detail),
                "type": "server_error" if e.status_code >= 500 else "invalid_request_error",
                "param": None,
                "code": "service_unavailable" if e.status_code >= 500 else "bad_request"
            }
        }
        return JSONResponse(content=error_response, status_code=e.status_code)
    except Exception as e:
        logger.exception(f"Unexpected error in images_generations endpoint: {e}")
        error_response = {
            "error": {
                "message": "An unexpected error occurred",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }
        return JSONResponse(content=error_response, status_code=500)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxies chat completion requests to the backend defined by the model's handle,
    using the same host as the MODEL_LIST_URL and the same OAuth token.
    """
    try:
        # 1. Read the original request data first
        original_request_data = await request.json()
        logger.debug(f"Received raw request data for chat completion: {original_request_data}")

        # 2. Determine streaming from the original request *before* any transformation
        is_streaming = original_request_data.get("stream", False)

        # 3. Make a copy to modify for the backend request payload
        payload_for_backend = original_request_data.copy()

        # Optional: Validate original request body structure against OpenAIChatCompletionRequest
        # try:
        #     OpenAIChatCompletionRequest.model_validate(original_request_data)
        # except Exception as e:
        #     logger.warning(f"Original request body validation failed: {e}")
        #     raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

        # 4. Perform model lookup using the model from the payload
        model_id = payload_for_backend.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field in request body")

        logger.info(f"Received chat completion request for model: {model_id}. Streaming: {is_streaming}")

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


        # --- API Specific Adjustments ---
        # Modify payload_for_backend if needed based on the target backend
        logger.debug(f"Checking handle '{handle}' for API specific adjustments.")
        if "/anthropic/" in handle.lower():
            # Adjust payload_for_backend in place for Anthropic
            if "max_tokens" not in payload_for_backend or payload_for_backend.get("max_tokens") is None:
                default_max_tokens = 4096 # Set a reasonable default for Anthropic
                logger.warning(f"Anthropic request for model '{model_id}' missing 'max_tokens'. Applying default: {default_max_tokens}")
                payload_for_backend["max_tokens"] = default_max_tokens
            # Add other Anthropic-specific transformations here if needed
            logger.debug(f"Anthropic payload for backend after adjustments for model '{model_id}': {payload_for_backend}")
        elif "/google/" in handle.lower(): # Check for /google/ instead of /gemini/
            # 5. Transform payload_for_backend for Gemini
            # This replaces the variable with the transformed dictionary
            payload_for_backend = transform_openai_request_to_gemini(payload_for_backend)
            logger.debug(f"Google/Gemini payload for backend after transformation for model '{model_id}': {payload_for_backend}")
            # Adjust target_url for Gemini streaming
            if is_streaming: # Use the flag determined from original request
                if target_url.endswith(":generateContent"):
                    # Replace non-streaming endpoint with streaming endpoint
                    target_url = target_url.replace(":generateContent", ":streamGenerateContent")
                    logger.info(f"Using Gemini streaming endpoint (replaced): {target_url}")
                elif ":streamGenerateContent" not in target_url:
                    # If it's not the standard non-streaming endpoint, try appending ?alt=sse as a fallback
                    logger.warning(f"Gemini handle '{handle}' does not end with ':generateContent'. Appending '?alt=sse' as fallback for streaming.")
                    if "?" in target_url:
                        target_url += "&alt=sse"
                    else:
                        target_url += "?alt=sse"
                    logger.info(f"Using Gemini streaming endpoint (fallback): {target_url}")
                else:
                     # Handle already points to a streaming endpoint
                     logger.info(f"Using Gemini streaming endpoint (pre-defined): {target_url}")

            # Note: Non-streaming Gemini requests use the original handle (e.g., :generateContent)
        elif "/cohere/" in handle.lower():
            # Transform payload for Cohere
            payload_for_backend = transform_openai_request_to_cohere(payload_for_backend)
            logger.debug(f"Cohere payload for backend after transformation for model '{model_id}': {payload_for_backend}")


        # --- Forward the request ---
        logger.info(f"Forwarding request for model '{model_id}' to {target_url}. Streaming: {is_streaming}")

        # Define the stream generator here, accepting necessary parameters
        async def stream_generator(
            req_url: str,
            backend_payload: dict, # Renamed parameter for clarity
            req_headers: dict,
            target_handle: str,
            requested_model: str
        ) -> AsyncGenerator[str, None]:
            """
            Streams response from backend. If the backend is Anthropic,
            transforms SSE events to OpenAI format. Otherwise, streams raw chunks.
            Yields strings formatted as SSE messages ('data: ...\n\n').
            """
            is_anthropic = "/anthropic/" in target_handle.lower()
            # Check for both /gemini/ and /google/ patterns
            is_gemini = "/gemini/" in target_handle.lower() or "/google/" in target_handle.lower()
            # Check for Cohere
            is_cohere = "/cohere/" in target_handle.lower()

            # Generate a base ID for the stream, potentially vendor-specific
            if is_anthropic:
                openai_chunk_id = f"chatcmpl-anthropic-{int(time.time())}"
            elif is_gemini:
                openai_chunk_id = f"chatcmpl-gemini-{int(time.time())}"
            elif is_cohere:
                openai_chunk_id = f"chatcmpl-cohere-{int(time.time())}"
            else: # Default/OpenAI compatible
                openai_chunk_id = f"chatcmpl-proxy-{int(time.time())}"


            async with httpx.AsyncClient() as stream_client:
                try:
                    # Gemini streaming uses GET for some models/versions, POST for others.
                    # Assuming POST based on typical generative AI patterns. Adjust if needed.
                    # Also, Gemini streaming URL already has ?alt=sse appended earlier.
                    request_method = "POST"
                    # Example: if target_handle == "/gemini/v1beta/models/gemini-pro:streamGenerateContent":
                    #     request_method = "POST" # Or GET if required by specific Gemini endpoint

                    # Log the actual data being sent for streaming requests
                    log_payload = backend_payload # Use a consistent variable name
                    if is_gemini:
                        logger.info(f"Sending streaming request for Gemini model '{requested_model}' to {req_url}")
                        logger.debug(f"Gemini stream request payload being sent to {req_url}: {log_payload}")
                    elif is_anthropic:
                         logger.info(f"Sending streaming request for Anthropic model '{requested_model}' to {req_url}")
                         logger.debug(f"Anthropic stream request payload being sent to {req_url}: {log_payload}")
                    else:
                         logger.info(f"Sending streaming request for model '{requested_model}' to {req_url}")
                         logger.debug(f"Stream request payload being sent to {req_url}: {log_payload}")


                    async with stream_client.stream(
                        request_method, req_url, json=log_payload, headers=req_headers, timeout=180.0
                    ) as backend_response:
                        logger.info(f"Stream connection established for model '{requested_model}' to {req_url}. Status: {backend_response.status_code}")
                        # Check for backend errors *before* streaming body
                        if backend_response.status_code >= 400:
                            error_body = await backend_response.aread()
                            try:
                                # Try to parse as JSON
                                error_json = json.loads(error_body.decode())

                                # Check if it's already in OpenAI format
                                if "error" in error_json and isinstance(error_json["error"], dict):
                                    error_detail = error_json
                                else:
                                    # Convert to OpenAI format
                                    error_message = error_json.get("detail", error_json.get("message", str(error_json)))
                                    if isinstance(error_message, dict):
                                        error_message = json.dumps(error_message)

                                    error_detail = {
                                        "error": {
                                            "message": error_message,
                                            "type": "server_error" if backend_response.status_code >= 500 else "invalid_request_error",
                                            "param": None,
                                            "code": "service_unavailable" if backend_response.status_code >= 500 else "bad_request"
                                        }
                                    }
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # Not JSON, use text
                                error_text = error_body.decode(errors='replace') or f"Backend error {backend_response.status_code}"
                                error_detail = {
                                    "error": {
                                        "message": error_text,
                                        "type": "server_error" if backend_response.status_code >= 500 else "invalid_request_error",
                                        "param": None,
                                        "code": "service_unavailable" if backend_response.status_code >= 500 else "bad_request"
                                    }
                                }

                            logger.error(f"Backend streaming request failed with status {backend_response.status_code}: {error_detail}")

                            # Yield the error in OpenAI format and then [DONE]
                            yield f"data: {json.dumps(error_detail)}\n\n"
                            yield "data: [DONE]\n\n"
                            return  # Stop the generator

                        logger.info(f"Starting stream processing for model '{requested_model}'. Backend type: {'Anthropic' if is_anthropic else 'Gemini' if is_gemini else 'Other'}")

                        if is_anthropic:
                            # Process Anthropic SSE stream line by line
                            logger.debug(f"Starting Anthropic SSE processing loop for model '{requested_model}'.")
                            current_event = None
                            current_data_lines = []
                            async for line in backend_response.aiter_lines():
                                logger.debug(f"Raw Anthropic SSE line: {line}")
                                if line.startswith("event:"):
                                    current_event = line[len("event:"):].strip()
                                    current_data_lines = [] # Reset data for new event
                                elif line.startswith("data:"):
                                    current_data_lines.append(line[len("data:"):].strip())
                                elif line == "": # Empty line signifies end of an event
                                    if current_event and current_data_lines:
                                        data_str = "".join(current_data_lines)
                                        try:
                                            event_data = json.loads(data_str)
                                            logger.debug(f"Parsed Anthropic event: {current_event}, data: {event_data}")

                                            openai_chunk = transform_anthropic_stream_chunk_to_openai(
                                                current_event, event_data, openai_chunk_id, requested_model
                                            )

                                            if openai_chunk:
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                                logger.debug(f"Yielded transformed OpenAI chunk: {openai_chunk}")

                                        except json.JSONDecodeError:
                                            logger.error(f"Failed to decode JSON data for event {current_event}: {data_str}")
                                        except Exception as transform_err:
                                             logger.exception(f"Error transforming Anthropic chunk: {transform_err}")

                                    # Reset for next event
                                    current_event = None
                                    current_data_lines = []
                            # After the loop, Anthropic stream is done
                            logger.info(f"Anthropic stream finished for model '{requested_model}'.")

                        elif is_gemini:
                             # Process Gemini SSE stream (JSON objects per line)
                             logger.debug(f"Starting Gemini SSE processing loop for model '{requested_model}'.")
                             async for line in backend_response.aiter_lines():
                                 logger.debug(f"Raw Gemini SSE line: {line}")
                                 # Gemini streams JSON objects, sometimes prefixed with "data: "
                                 if line.startswith("data: "):
                                     line = line[len("data: "):]
                                 line = line.strip()
                                 if not line:
                                     continue # Skip empty lines (often act as SSE message separators)

                                 try:
                                     # The actual payload is the JSON object itself
                                     gemini_chunk_data = json.loads(line)
                                     logger.debug(f"Parsed Gemini chunk data: {gemini_chunk_data}")

                                     openai_chunk = transform_gemini_stream_chunk_to_openai(
                                         gemini_chunk_data, openai_chunk_id, requested_model
                                     )

                                     if openai_chunk:
                                         yield f"data: {json.dumps(openai_chunk)}\n\n"
                                         logger.debug(f"Yielded transformed OpenAI chunk: {openai_chunk}")
                                     # else: Chunk transformation resulted in None (e.g., empty delta), do nothing

                                 except json.JSONDecodeError:
                                     logger.error(f"Failed to decode JSON data from Gemini stream line: {line}")
                                 except Exception as transform_err:
                                     logger.exception(f"Error transforming Gemini chunk: {transform_err}")
                             logger.info(f"Gemini stream finished for model '{requested_model}'.")

                        elif is_cohere:
                             # Process Cohere SSE stream
                             logger.debug(f"Starting Cohere SSE processing loop for model '{requested_model}'.")
                             async for line in backend_response.aiter_lines():
                                 logger.debug(f"Raw Cohere SSE line: {line}")
                                 # Cohere streams JSON objects, prefixed with "data: "
                                 if line.startswith("data: "):
                                     line = line[len("data: "):]
                                 line = line.strip()
                                 if not line or line == "[DONE]":
                                     # Skip empty lines or Cohere's own [DONE] marker
                                     if line == "[DONE]":
                                         logger.debug("Received Cohere [DONE] marker")
                                     continue

                                 try:
                                     # Parse the JSON chunk
                                     cohere_chunk_data = json.loads(line)
                                     logger.debug(f"Parsed Cohere chunk data: {cohere_chunk_data}")

                                     # Transform to OpenAI format
                                     openai_chunk = transform_cohere_stream_chunk_to_openai(
                                         cohere_chunk_data, openai_chunk_id, requested_model
                                     )

                                     if openai_chunk:
                                         yield f"data: {json.dumps(openai_chunk)}\n\n"
                                         logger.debug(f"Yielded transformed OpenAI chunk: {openai_chunk}")

                                 except json.JSONDecodeError:
                                     logger.error(f"Failed to decode JSON data from Cohere stream line: {line}")
                                 except Exception as transform_err:
                                     logger.exception(f"Error transforming Cohere chunk: {transform_err}")
                             logger.info(f"Cohere stream finished for model '{requested_model}'.")

                        else: # Not Anthropic, Gemini, or Cohere, assume OpenAI compatible stream (forward raw lines)
                            logger.debug(f"Starting raw SSE forwarding loop for model '{requested_model}'.")
                            # Note: This assumes the non-Anthropic/non-Gemini backend ALREADY sends OpenAI formatted SSE.
                            # Iterate line by line to ensure proper SSE formatting.
                            async for line in backend_response.aiter_lines():
                                logger.debug(f"Raw backend SSE line: {line}")
                                # Forward lines directly, adding the necessary SSE line endings
                                yield f"{line}\n" # Forward the line itself, adding newline for SSE
                            logger.info(f"Finished forwarding raw stream from backend for model '{requested_model}'")

                        # Send the final [DONE] message for all streams ONLY if not handled by backend
                        # Note: OpenAI backends usually send their own [DONE] message.
                        # Anthropic transformation adds its own [DONE].
                        # Forwarding raw chunks might include the backend's [DONE].
                        # Send the final [DONE] message for transformed streams
                        if is_anthropic or is_gemini:
                             yield "data: [DONE]\n\n"
                             logger.info(f"Sent [DONE] message for model '{requested_model}'.")
                        # For raw forwarding, assume the backend sends its own [DONE] if needed.

                except HTTPException:
                     raise # Re-raise HTTP exceptions from status check
                except Exception as e:
                    # Log errors occurring during the streaming process itself
                    logger.exception(f"Error during backend stream processing for model {requested_model}: {e}")
                    # Format error in OpenAI-compatible format
                    error_payload = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "param": None,
                            "code": "service_unavailable"
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n" # Send DONE after error per OpenAI spec


        # Now, handle the request based on the is_streaming flag determined earlier
        try:
            if is_streaming:
                # Log the data being passed to the generator
                logger.debug(f"Passing payload to stream_generator: {payload_for_backend}")
                logger.debug(f"Raw request body for streaming: {payload_for_backend}")
                # Create the generator instance, passing the potentially transformed payload
                generator = stream_generator(target_url, payload_for_backend, headers_to_forward, handle, model_id)
                # Return a StreamingResponse using the generator
                # OpenAI standard content type for streaming is text/event-stream
                return StreamingResponse(generator, media_type="text/event-stream")

            else: # Non-streaming request
                # Use a separate client instance for non-streaming requests
                logger.info(f"Sending non-streaming request for model '{model_id}' to {target_url}")
                logger.debug(f"Non-streaming payload for '{model_id}': {payload_for_backend}")
                logger.debug(f"Raw request body for non-streaming: {payload_for_backend}")
                async with httpx.AsyncClient() as client:
                    backend_response = await client.post(
                        target_url,
                        json=payload_for_backend, # Send the potentially transformed payload
                        headers=headers_to_forward,
                        timeout=180.0 # Set a reasonable timeout for LLM requests
                    )

                    # Raise exception for 4xx/5xx responses from the backend
                    backend_response.raise_for_status()

                    # Return the raw JSON response from the backend
                    response_data = backend_response.json()
                    logger.info(f"Successfully received non-streaming response from backend for model '{model_id}'")
                    logger.debug(f"Backend response data for model '{model_id}': {response_data}")
                    logger.debug(f"Raw request body: {payload_for_backend}")
                    logger.debug(f"Raw response body: {backend_response.text}")

                    # --- Transform response if needed ---
                    final_response_data = response_data
                    # Check if this is an OpenAI response with a nested 'response' field
                    if "/openai/" in handle.lower() and "response" in response_data:
                        logger.info(f"Extracting nested response for OpenAI model '{model_id}'")
                        final_response_data = response_data.get("response", {})
                    elif "/anthropic/" in handle.lower():
                        logger.info(f"Transforming Anthropic response for model '{model_id}' to OpenAI format.")
                        final_response_data = transform_anthropic_response_to_openai(response_data, model_id)
                        # Check if transformation resulted in an error structure
                        if "error" in final_response_data:
                             # Format as OpenAI-compatible error
                             error_message = final_response_data.get("error", {}).get("message", "Transformation error")
                             logger.error(f"Anthropic transformation failed: {error_message}")
                             error_response = {
                                 "error": {
                                     "message": error_message,
                                     "type": "server_error",
                                     "param": None,
                                     "code": "internal_error"
                                 }
                             }
                             return JSONResponse(content=error_response, status_code=500)
                    # Check for both /gemini/ and /google/ patterns for response transformation
                    elif "/gemini/" in handle.lower() or "/google/" in handle.lower():
                        logger.info(f"Transforming Gemini/Google response for model '{model_id}' to OpenAI format.")
                        final_response_data = transform_gemini_response_to_openai(response_data.get('response', {}), model_id)
                        if "error" in final_response_data:
                             # Format as OpenAI-compatible error
                             error_message = final_response_data.get("error", {}).get("message", "Transformation error")
                             logger.error(f"Gemini transformation failed: {error_message}")
                             error_response = {
                                 "error": {
                                     "message": error_message,
                                     "type": "server_error",
                                     "param": None,
                                     "code": "internal_error"
                                 }
                             }
                             return JSONResponse(content=error_response, status_code=500)
                    # Check for Cohere
                    elif "/cohere/" in handle.lower():
                        logger.info(f"Transforming Cohere response for model '{model_id}' to OpenAI format.")
                        final_response_data = transform_cohere_response_to_openai(response_data, model_id)
                        if "error" in final_response_data:
                             # Format as OpenAI-compatible error
                             error_message = final_response_data.get("error", {}).get("message", "Transformation error")
                             logger.error(f"Cohere transformation failed: {error_message}")
                             error_response = {
                                 "error": {
                                     "message": error_message,
                                     "type": "server_error",
                                     "param": None,
                                     "code": "internal_error"
                                 }
                             }
                             return JSONResponse(content=error_response, status_code=500)


                    # Return potentially transformed data with original success status code
                    return JSONResponse(content=final_response_data, status_code=backend_response.status_code)

        except httpx.RequestError as e:
            logger.error(f"Error requesting backend {target_url}: {e}")
            # Format as OpenAI-compatible error
            error_response = {
                "error": {
                    "message": f"Error contacting backend service: {e}",
                    "type": "server_error",
                    "param": None,
                    "code": "service_unavailable"
                }
            }
            return JSONResponse(content=error_response, status_code=503)
        except httpx.HTTPStatusError as e: # Caught for non-streaming errors
             logger.error(f"Backend service at {target_url} returned error {e.response.status_code}: {e.response.text}")
             try:
                 # Try to parse backend error response
                 error_content = e.response.json()

                 # Check if it's already in OpenAI format
                 if "error" in error_content and isinstance(error_content["error"], dict):
                     # Already in OpenAI format, pass through
                     return JSONResponse(content=error_content, status_code=e.response.status_code)
                 else:
                     # Convert to OpenAI format
                     error_message = error_content.get("detail", error_content.get("message", str(error_content)))
                     if isinstance(error_message, dict):
                         error_message = json.dumps(error_message)

                     error_response = {
                         "error": {
                             "message": error_message,
                             "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                             "param": None,
                             "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                         }
                     }
                     return JSONResponse(content=error_response, status_code=e.response.status_code)
             except Exception:
                 # Fallback if response is not JSON
                 error_response = {
                     "error": {
                         "message": e.response.text or f"Backend error {e.response.status_code}",
                         "type": "server_error" if e.response.status_code >= 500 else "invalid_request_error",
                         "param": None,
                         "code": "service_unavailable" if e.response.status_code >= 500 else "bad_request"
                     }
                 }
                 return JSONResponse(content=error_response, status_code=e.response.status_code)

    except HTTPException as e:
         # Convert FastAPI HTTPException to OpenAI-compatible error format
         logger.error(f"HTTP exception in chat_completions: {e.detail} (status: {e.status_code})")
         error_response = {
             "error": {
                 "message": str(e.detail),
                 "type": "server_error" if e.status_code >= 500 else "invalid_request_error",
                 "param": None,
                 "code": "service_unavailable" if e.status_code >= 500 else "bad_request"
             }
         }
         return JSONResponse(content=error_response, status_code=e.status_code)
    except Exception as e:
        logger.exception(f"Unexpected error in chat_completions endpoint: {e}") # Log full traceback
        error_response = {
            "error": {
                "message": "An unexpected error occurred",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }
        return JSONResponse(content=error_response, status_code=500)


# --- Helper Functions ---
def transform_anthropic_response_to_openai(anthropic_data: Dict[str, Any], requested_model_id: str) -> Dict[str, Any]:
    """
    Transforms a non-streaming Anthropic API response (/v1/messages) into the
    OpenAI Chat Completion format.
    """
    logger.debug(f"Attempting to transform Anthropic response for model {requested_model_id}")
    try:
        # The actual response payload seems nested under 'response' key based on logs
        anthropic_response = anthropic_data.get("response", {})
        if not anthropic_response:
            logger.error("Anthropic response data is missing the 'response' field.")
            # Return a generic error structure or raise? Let's return error structure.
            return {
                "error": {
                    "message": "Invalid response format received from Anthropic backend (missing 'response' field).",
                    "type": "server_error",
                    "param": None,
                    "code": "invalid_response_error"
                }
            }

        anthropic_id = anthropic_response.get("id", f"anthropic-id-{int(time.time())}")
        # anthropic_model = anthropic_response.get("model", requested_model_id) # Use actual model if available, else requested
        anthropic_usage = anthropic_response.get("usage", {})
        anthropic_content = anthropic_response.get("content", [])
        anthropic_role = anthropic_response.get("role", "assistant")
        anthropic_stop_reason = anthropic_response.get("stop_reason")

        # --- Map Content ---
        # Assuming the first content block is the primary text response
        message_content = ""
        if anthropic_content and isinstance(anthropic_content, list) and len(anthropic_content) > 0:
            first_content_block = anthropic_content[0]
            if first_content_block.get("type") == "text":
                message_content = first_content_block.get("text", "")

        # --- Map Finish Reason ---
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            # Add other mappings if needed: tool_use?
        }
        finish_reason = finish_reason_map.get(anthropic_stop_reason, anthropic_stop_reason) # Default to original if no map

        # --- Map Usage ---
        prompt_tokens = anthropic_usage.get("input_tokens", 0)
        completion_tokens = anthropic_usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # --- Construct OpenAI Response ---
        openai_response = {
            "id": f"chatcmpl-{anthropic_id}", # Prefix Anthropic ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model_id, # Use the model ID requested by the client, per OpenAI spec
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": anthropic_role,
                        "content": message_content,
                    },
                    "finish_reason": finish_reason,
                    "logprobs": None, # Anthropic doesn't provide logprobs in this basic response
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "system_fingerprint": None, # Anthropic doesn't provide this
            # Include other fields if they can be mapped
        }
        logger.debug(f"Successfully transformed Anthropic response to OpenAI format.")
        return openai_response

    except Exception as e:
        logger.exception(f"Error transforming Anthropic non-streaming response: {e}. Original data: {anthropic_data}")
        # Return a generic error structure
        return {
            "error": {
                "message": f"Error transforming response from Anthropic backend: {e}",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }


def transform_anthropic_stream_chunk_to_openai(
    event_type: str, event_data: Dict[str, Any], chunk_id: str, model_id: str
) -> Optional[Dict[str, Any]]:
    """
    Transforms a parsed Anthropic SSE event (type and data) into an OpenAI
    Chat Completion Chunk dictionary. Returns None if the event doesn't map.
    """
    logger.debug(f"Transforming Anthropic event: {event_type}")
    openai_chunk = {
        "id": chunk_id, # Use the consistent ID for all chunks in the stream
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {}, # Populated based on event type
            "finish_reason": None,
            "logprobs": None,
        }],
        "usage": None, # Anthropic sends usage in message_delta or message_stop
        "system_fingerprint": None, # Not provided by Anthropic
    }

    try:
        if event_type == "message_start":
            # message_start contains the initial message structure, including role
            # We can potentially extract the role for the first delta chunk.
            role = event_data.get("message", {}).get("role", "assistant")
            openai_chunk["choices"][0]["delta"] = {"role": role}
            # Anthropic also sends usage here, but OpenAI expects it at the end.
            # We could store it and add it to the *last* chunk if needed, but
            # OpenAI clients usually ignore usage in intermediate chunks.

        elif event_type == "content_block_delta":
            # This event contains the actual text changes
            delta_info = event_data.get("delta", {})
            if delta_info.get("type") == "text_delta":
                text_delta = delta_info.get("text", "")
                if text_delta: # Only include content if there is text
                     openai_chunk["choices"][0]["delta"] = {"content": text_delta}
                else:
                     return None # No actual content change, skip chunk

        elif event_type == "message_delta":
            # This event signals the end of the message and provides stop reason/usage
            delta_info = event_data.get("delta", {})
            stop_reason = delta_info.get("stop_reason")
            usage_update = event_data.get("usage", {}) # Contains output_tokens

            if stop_reason:
                 # Map Anthropic stop reasons to OpenAI finish reasons
                 finish_reason_map = {
                     "end_turn": "stop",
                     "max_tokens": "length",
                     "stop_sequence": "stop",
                     # Add other mappings if needed: tool_use?
                 }
                 finish_reason = finish_reason_map.get(stop_reason, stop_reason)
                 openai_chunk["choices"][0]["finish_reason"] = finish_reason
                 openai_chunk["choices"][0]["delta"] = {} # Delta is empty for the final chunk

                 # Add usage info if available (OpenAI spec includes it in the *last* chunk)
                 # Note: Anthropic provides output tokens here, but not input tokens.
                 # Input tokens are in message_start. Combining them accurately in
                 # the final chunk is complex. For now, let's omit usage from streaming.
                 # if usage_update.get("output_tokens"):
                 #    openai_chunk["usage"] = {"completion_tokens": usage_update["output_tokens"]}
                 #    # We'd need to have stored prompt_tokens from message_start to add total_tokens

            else:
                 # message_delta might occur without stop_reason (e.g., intermediate updates)
                 # In this case, it doesn't directly map to an OpenAI chunk, so skip.
                 return None

        elif event_type == "message_stop":
            # This confirms the stream end from Anthropic's side.
            # OpenAI uses `data: [DONE]\n\n`, which is handled outside this function.
            # We might extract final usage here if needed and not captured in message_delta.
            return None # No direct OpenAI chunk equivalent

        elif event_type == "ping":
            # Keepalive event, ignore.
            return None

        else:
            # Unknown event type
            logger.warning(f"Unhandled Anthropic SSE event type: {event_type}")
            return None

        # Only return the chunk if delta is not empty or finish_reason is set
        if openai_chunk["choices"][0]["delta"] or openai_chunk["choices"][0]["finish_reason"]:
            return openai_chunk
        else:
            # Avoid sending empty chunks if no delta/finish_reason was populated
            return None

    except Exception as e:
        logger.exception(f"Error transforming Anthropic stream chunk: {e}. Event: {event_type}, Data: {event_data}")
        return None # Skip chunk on error


# --- Gemini Transformation Functions ---

def transform_openai_request_to_gemini(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms an OpenAI Chat Completion request to a Gemini API request."""
    logger.debug("Transforming OpenAI request to Gemini format.")
    gemini_request = {"contents": []}
    generation_config = {}
    safety_settings = [] # Optional: Add default safety settings if needed

    # Map messages to contents
    system_prompt = None
    for message in openai_data.get("messages", []):
        role = message.get("role")
        content = message.get("content")

        if not content: # Skip messages without content
            continue

        # Handle system prompt (Gemini prefers it at the start or via specific field)
        # Simple approach: Prepend system prompt to the first user message content.
        # More robust: Use Gemini's 'system_instruction' field if available/supported by endpoint.
        if role == "system":
            system_prompt = content
            continue # Don't add system message directly to contents yet

        # Map roles
        gemini_role = "user" # Default
        if role == "assistant":
            gemini_role = "model"
        elif role == "tool":
            # TODO: Handle tool calls/results if needed for Gemini
            logger.warning("Gemini transformation: Tool messages not fully handled yet.")
            continue # Skip tool messages for now

        # Combine system prompt with the first user message
        if gemini_role == "user" and system_prompt:
            content = f"{system_prompt}\n\n{content}"
            system_prompt = None # Only add it once

        # Gemini expects content as {"parts": [{"text": "..."}]}
        gemini_request["contents"].append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })

    # Handle case where only a system prompt was provided (maybe less common)
    if system_prompt and not gemini_request["contents"]:
         gemini_request["contents"].append({
             "role": "user", # Treat standalone system prompt as initial user message
             "parts": [{"text": system_prompt}]
         })

    # Map generation parameters
    if openai_data.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = openai_data["max_tokens"]
    if openai_data.get("temperature") is not None:
        generation_config["temperature"] = openai_data["temperature"]
    if openai_data.get("top_p") is not None:
        generation_config["topP"] = openai_data["top_p"]
    if openai_data.get("stop"):
        # Ensure stop sequences are strings
        stop_sequences = [str(s) for s in openai_data["stop"]]
        generation_config["stopSequences"] = stop_sequences
    # Gemini doesn't directly support 'n', 'presence_penalty', 'frequency_penalty', 'logit_bias' in the same way
    if openai_data.get("n", 1) > 1:
        logger.warning("Gemini transformation: 'n > 1' is not directly supported. Requesting only one candidate.")
        generation_config["candidateCount"] = 1 # Explicitly set to 1, though often the default
    if any(k in openai_data for k in ["presence_penalty", "frequency_penalty", "logit_bias"]):
        logger.warning("Gemini transformation: 'presence_penalty', 'frequency_penalty', 'logit_bias' are not supported.")


    # Add generationConfig if not empty
    if generation_config:
        gemini_request["generationConfig"] = generation_config

    # Add safetySettings if configured
    if safety_settings:
        gemini_request["safetySettings"] = safety_settings

    # Remove OpenAI-specific fields not used by Gemini (like 'stream', 'model')
    # 'stream' is handled via URL param (?alt=sse)
    # 'model' is implicit in the endpoint URL

    logger.debug(f"Transformed Gemini request: {gemini_request}")
    return gemini_request


def transform_gemini_response_to_openai(gemini_data: Dict[str, Any], requested_model_id: str) -> Dict[str, Any]:
    """Transforms a non-streaming Gemini API response into the OpenAI Chat Completion format."""
    logger.debug(f"Attempting to transform Gemini response for model {requested_model_id}")
    try:
        # Gemini response structure: { "candidates": [...], "usageMetadata": {...} }
        candidates = gemini_data.get("candidates", [])
        usage_metadata = gemini_data.get("usageMetadata", {})

        if not candidates:
            logger.error("Gemini response data is missing 'candidates' field.")
            return {
                "error": {
                    "message": "Invalid response format received from Gemini backend (missing 'candidates').",
                    "type": "server_error",
                    "param": None,
                    "code": "invalid_response_error"
                }
            }

        # Process the first candidate
        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        message_content = ""
        if content.get("role") == "model" and content.get("parts"):
            # Concatenate text from all parts (usually just one)
            message_content = "".join(part.get("text", "") for part in content["parts"] if "text" in part)

        # Map finish reason
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter", # OpenAI uses 'content_filter'
            "RECITATION": "stop", # Or maybe a custom reason? Mapping to 'stop' for now.
            "OTHER": "stop", # Generic stop
            "FINISH_REASON_UNSPECIFIED": None, # Map unspecified to None
        }
        gemini_finish_reason = first_candidate.get("finishReason")
        finish_reason = finish_reason_map.get(gemini_finish_reason, None) # Default to None if unknown/unspecified

        # Map usage
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0) # Sum across candidates if needed
        total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens) # Use total if available

        # Construct OpenAI Response
        openai_response = {
            "id": f"chatcmpl-gemini-{int(time.time())}", # Generate an ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model_id, # Use the model ID requested by the client
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant", # Gemini response is always 'model' role
                        "content": message_content,
                    },
                    "finish_reason": finish_reason,
                    "logprobs": None, # Gemini API doesn't provide logprobs here
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "system_fingerprint": None, # Gemini doesn't provide this
        }
        logger.debug("Successfully transformed Gemini response to OpenAI format.")
        return openai_response

    except Exception as e:
        logger.exception(f"Error transforming Gemini non-streaming response: {e}. Original data: {gemini_data}")
        return {
            "error": {
                "message": f"Error transforming response from Gemini backend: {e}",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }


def transform_gemini_stream_chunk_to_openai(
    gemini_chunk: Dict[str, Any], chunk_id: str, model_id: str
) -> Optional[Dict[str, Any]]:
    """
    Transforms a Gemini SSE chunk (parsed JSON object) into an OpenAI
    Chat Completion Chunk dictionary. Returns None if the chunk doesn't map.
    """
    # Gemini chunk structure is similar to the full response but incremental:
    # { "candidates": [ { "content": { "role": "model", "parts": [ { "text": "..." } ] }, "finishReason": ..., "index": 0 } ], "usageMetadata": { ... } }
    # Sometimes only delta is present in candidates[0].content.parts[0].text
    logger.debug(f"Transforming Gemini chunk: {gemini_chunk}")
    openai_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": None,
            "logprobs": None,
        }],
        "usage": None, # Usage might appear in the last chunk's usageMetadata
        "system_fingerprint": None,
    }

    try:
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            # Sometimes Gemini sends empty chunks or just usage metadata
            # Check for usage metadata in the chunk itself
            usage_metadata = gemini_chunk.get("usageMetadata")
            if usage_metadata:
                 # This might be the final chunk containing usage. OpenAI spec is a bit ambiguous
                 # on whether usage should be in the *last* delta chunk or a separate final message.
                 # Let's try adding it to the last chunk with a finish_reason if possible,
                 # but for now, we'll ignore usage updates in streaming chunks.
                 logger.debug(f"Gemini chunk contains usageMetadata: {usage_metadata}. Ignoring for now in streaming.")
                 return None # Don't send a chunk just for usage metadata yet.
            else:
                 logger.warning("Received Gemini chunk without candidates or usage metadata.")
                 return None # Skip empty or unrecognized chunks


        first_candidate = candidates[0]
        delta_content = ""
        delta_role = None # Role usually appears only once at the start

        # Extract content delta
        content = first_candidate.get("content", {})
        if content.get("role") == "model":
            delta_role = "assistant" # Map role if present
        if content.get("parts"):
            delta_content = "".join(part.get("text", "") for part in content["parts"] if "text" in part)

        # Populate delta
        if delta_role and delta_content: # First chunk might have both role and content
             openai_chunk["choices"][0]["delta"] = {"role": delta_role, "content": delta_content}
        elif delta_role: # First chunk might only have role
             openai_chunk["choices"][0]["delta"] = {"role": delta_role}
        elif delta_content: # Subsequent chunks usually only have content
             openai_chunk["choices"][0]["delta"] = {"content": delta_content}
        else:
             # If no role or content delta, check for finish reason
             pass # Continue to check finish reason

        # Extract finish reason
        gemini_finish_reason = first_candidate.get("finishReason")
        if gemini_finish_reason and gemini_finish_reason != "FINISH_REASON_UNSPECIFIED":
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "stop",
                "OTHER": "stop",
            }
            finish_reason = finish_reason_map.get(gemini_finish_reason)
            openai_chunk["choices"][0]["finish_reason"] = finish_reason
            # Ensure delta is empty if only finish_reason is present in this chunk
            if not delta_content and not delta_role:
                 openai_chunk["choices"][0]["delta"] = {}


        # Only return the chunk if it contains a delta or a finish reason
        if openai_chunk["choices"][0]["delta"] or openai_chunk["choices"][0]["finish_reason"] is not None:
            # Avoid sending delta: {} if only finish_reason is set
            if not openai_chunk["choices"][0]["delta"] and openai_chunk["choices"][0]["finish_reason"] is not None:
                 openai_chunk["choices"][0]["delta"] = {} # Ensure delta field exists but is empty
            elif not openai_chunk["choices"][0]["delta"]: # Should not happen if finish_reason is None, but safety check
                 return None

            return openai_chunk
        else:
            logger.debug("Skipping Gemini chunk transformation as it resulted in no delta or finish reason.")
            return None # Skip chunk if it resulted in no meaningful update

    except Exception as e:
        logger.exception(f"Error transforming Gemini stream chunk: {e}. Chunk: {gemini_chunk}")
        return None # Skip chunk on error


# --- Cohere Transformation Functions ---

def transform_openai_request_to_cohere(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms an OpenAI Chat Completion request to a Cohere API request."""
    logger.debug("Transforming OpenAI request to Cohere format.")

    # Initialize Cohere request structure with the correct fields
    # Based on Cohere API documentation and testing
    cohere_request = {
        "messages": [],
        "model": openai_data.get("model", "command-r")
    }

    # Extract messages from OpenAI request
    openai_messages = openai_data.get("messages", [])

    # Process system message if present
    system_prompt = None
    for msg in openai_messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break

    # Add system prompt as preamble if present
    if system_prompt:
        cohere_request["preamble"] = system_prompt

    # Convert OpenAI messages to Cohere format
    # Cohere expects an array of message objects with role and content
    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        # Skip system messages (already handled via preamble)
        if role == "system":
            continue

        # Map roles - Cohere uses "USER" and "CHATBOT" (uppercase)
        cohere_role = "user" if role == "user" else "assistant"

        # Add to messages array
        cohere_request["messages"].append({
            "role": cohere_role,
            "content": content
        })

    # Map other parameters
    if "temperature" in openai_data:
        cohere_request["temperature"] = openai_data["temperature"]

    if "top_p" in openai_data:
        cohere_request["p"] = openai_data["top_p"]

    if "max_tokens" in openai_data:
        cohere_request["max_tokens"] = openai_data["max_tokens"]

    if "stream" in openai_data:
        cohere_request["stream"] = openai_data["stream"]

    logger.debug(f"Transformed Cohere request: {cohere_request}")
    return cohere_request


def transform_cohere_response_to_openai(cohere_data: Dict[str, Any], requested_model_id: str) -> Dict[str, Any]:
    """Transforms a non-streaming Cohere API response to OpenAI Chat Completion format."""
    logger.debug(f"Transforming Cohere response for model {requested_model_id}")
    logger.debug(f"Raw Cohere response data: {cohere_data}")

    try:
        # Extract the response data
        # The actual response payload is nested under 'response' key
        cohere_response = cohere_data.get("response", {})
        if not cohere_response:
            logger.error("Cohere response data is missing the 'response' field.")
            return {
                "error": {
                    "message": "Invalid response format received from Cohere backend (missing 'response' field).",
                    "type": "server_error",
                    "param": None,
                    "code": "invalid_response_error"
                }
            }

        # Log the full response structure for debugging
        logger.debug(f"Cohere response structure: {cohere_response}")

        # Extract response components
        cohere_id = cohere_response.get("id", f"cohere-id-{int(time.time())}")

        # Get the text content from the response
        # Cohere API returns a message object with content field
        cohere_text = ""

        # Try to extract text from the message field
        if "message" in cohere_response:
            message = cohere_response.get("message", {})
            if isinstance(message, dict):
                # Check for content field which may be a string or an array
                if "content" in message:
                    content = message.get("content")
                    if isinstance(content, str):
                        cohere_text = content
                        logger.debug(f"Found text in message.content (string): {cohere_text[:100]}...")
                    elif isinstance(content, list):
                        # Handle content array format
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                cohere_text += item.get("text", "")
                        logger.debug(f"Found text in message.content[].text: {cohere_text[:100]}...")

        # If we still don't have text, try other possible locations
        if not cohere_text:
            # Try text field directly in the response
            if "text" in cohere_response:
                cohere_text = cohere_response.get("text", "")
                logger.debug(f"Found text directly in response: {cohere_text[:100]}...")
            # Try generations array
            elif "generations" in cohere_response and isinstance(cohere_response["generations"], list):
                generations = cohere_response["generations"]
                if generations and "text" in generations[0]:
                    cohere_text = generations[0]["text"]
                    logger.debug(f"Found text in generations[0]: {cohere_text[:100]}...")

        # If we still don't have text, search all fields as a last resort
        if not cohere_text:
            logger.warning("Could not find text in standard locations, searching all fields")
            for key, value in cohere_response.items():
                if isinstance(value, str) and len(value) > 10:
                    cohere_text = value
                    logger.debug(f"Found potential text in field '{key}': {cohere_text[:100]}...")
                    break

        # Extract finish reason
        cohere_finish_reason = cohere_response.get("finish_reason")

        # Extract usage information
        cohere_usage = None
        if "usage" in cohere_data:
            cohere_usage = cohere_data.get("usage", {})
        elif "usage" in cohere_response:
            cohere_usage = cohere_response.get("usage", {})

        # Map finish reason
        finish_reason_map = {
            "COMPLETE": "stop",
            "MAX_TOKENS": "length",
            "ERROR": "error",
            "CANCELLED": "stop",
            "SAFETY": "content_filter"
        }
        finish_reason = finish_reason_map.get(cohere_finish_reason, "stop")

        # Map usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if cohere_usage:
            logger.debug(f"Processing usage data: {cohere_usage}")
            # Try different possible locations for token counts
            if "input_tokens" in cohere_usage:
                prompt_tokens = cohere_usage.get("input_tokens", 0)
            elif "tokens" in cohere_usage and "input_tokens" in cohere_usage.get("tokens", {}):
                prompt_tokens = cohere_usage.get("tokens", {}).get("input_tokens", 0)
            elif "billed_units" in cohere_usage and "input_tokens" in cohere_usage.get("billed_units", {}):
                prompt_tokens = cohere_usage.get("billed_units", {}).get("input_tokens", 0)

            if "output_tokens" in cohere_usage:
                completion_tokens = cohere_usage.get("output_tokens", 0)
            elif "tokens" in cohere_usage and "output_tokens" in cohere_usage.get("tokens", {}):
                completion_tokens = cohere_usage.get("tokens", {}).get("output_tokens", 0)
            elif "billed_units" in cohere_usage and "output_tokens" in cohere_usage.get("billed_units", {}):
                completion_tokens = cohere_usage.get("billed_units", {}).get("output_tokens", 0)

            total_tokens = prompt_tokens + completion_tokens
            logger.debug(f"Extracted token counts - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

        # Process the text content to extract from code blocks if needed
        # This handles cases where Cohere returns markdown code blocks
        if cohere_text and "```" in cohere_text:
            # Check for JSON code blocks specifically
            import re
            json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', cohere_text, re.DOTALL)
            if json_block_match:
                # Extract the content inside the code block
                extracted_content = json_block_match.group(1).strip()
                logger.debug(f"Extracted content from code block: {extracted_content}")
                cohere_text = extracted_content
            else:
                # For other code blocks, just remove the markdown formatting
                cohere_text = re.sub(r'```.*?\n', '', cohere_text)
                cohere_text = cohere_text.replace('```', '')
                logger.debug(f"Removed code block formatting: {cohere_text}")

        # Construct OpenAI response
        openai_response = {
            "id": f"chatcmpl-{cohere_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": cohere_text,
                    },
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "system_fingerprint": None,
        }

        logger.debug(f"Successfully transformed Cohere response to OpenAI format: {openai_response}")
        return openai_response

    except Exception as e:
        logger.exception(f"Error transforming Cohere response: {e}. Original data: {cohere_data}")
        return {
            "error": {
                "message": f"Error transforming response from Cohere backend: {e}",
                "type": "server_error",
                "param": None,
                "code": "internal_error"
            }
        }


def transform_cohere_stream_chunk_to_openai(
    cohere_chunk: Dict[str, Any], chunk_id: str, model_id: str
) -> Optional[Dict[str, Any]]:
    """
    Transforms a Cohere streaming chunk into an OpenAI Chat Completion Chunk.
    Returns None if the chunk doesn't map to a valid OpenAI chunk.
    """
    logger.debug(f"Transforming Cohere stream chunk: {cohere_chunk}")

    openai_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": None,
            "logprobs": None,
        }],
        "usage": None,
        "system_fingerprint": None,
    }

    try:
        # Log the full chunk structure for debugging
        logger.debug(f"Cohere stream chunk structure: {json.dumps(cohere_chunk, default=str)}")

        # Check for event type (older format)
        event_type = cohere_chunk.get("event_type")
        
        # Check for type (newer format)
        chunk_type = cohere_chunk.get("type")

        # Handle newer Cohere streaming format
        if chunk_type == "message-start":
            # First chunk - send role
            openai_chunk["choices"][0]["delta"] = {"role": "assistant"}
            return openai_chunk
            
        elif chunk_type == "content-delta":
            # Content chunk in newer format
            # Extract text from the nested structure
            text = ""
            if "delta" in cohere_chunk and "message" in cohere_chunk["delta"]:
                message = cohere_chunk["delta"]["message"]
                if "content" in message:
                    content = message["content"]
                    if isinstance(content, dict) and "text" in content:
                        text = content["text"]
            
            if text:
                # Filter code block markers if present
                if "```" in text:
                    text = text.replace("```json", "").replace("```", "")
                
                openai_chunk["choices"][0]["delta"] = {"content": text}
                return openai_chunk
            else:
                return None  # Skip empty content
                
        elif chunk_type == "message-end":
            # Final chunk with finish reason
            finish_reason_map = {
                "COMPLETE": "stop",
                "MAX_TOKENS": "length",
                "ERROR": "error",
                "CANCELLED": "stop",
                "SAFETY": "content_filter"
            }
            
            finish_reason = None
            usage_data = None
            
            if "delta" in cohere_chunk:
                delta = cohere_chunk["delta"]
                finish_reason = delta.get("finish_reason")
                usage_data = delta.get("usage")
            
            if finish_reason:
                mapped_reason = finish_reason_map.get(finish_reason, "stop")
                openai_chunk["choices"][0]["delta"] = {}
                openai_chunk["choices"][0]["finish_reason"] = mapped_reason
                
                # Add usage if available
                if usage_data:
                    openai_chunk["usage"] = {
                        "prompt_tokens": usage_data.get("tokens", {}).get("input_tokens", 0),
                        "completion_tokens": usage_data.get("tokens", {}).get("output_tokens", 0),
                        "total_tokens": (
                            usage_data.get("tokens", {}).get("input_tokens", 0) + 
                            usage_data.get("tokens", {}).get("output_tokens", 0)
                        )
                    }
                
                return openai_chunk
            else:
                return None
                
        # Handle older Cohere streaming format
        elif event_type == "stream-start":
            # First chunk - send role
            openai_chunk["choices"][0]["delta"] = {"role": "assistant"}
            return openai_chunk

        elif event_type == "text-generation":
            # Content chunk
            text = cohere_chunk.get("text", "")
            if text:
                # Check if this chunk contains markdown code block markers
                if "```" in text:
                    # Replace code block markers with empty string
                    text = text.replace("```json", "").replace("```", "")
                    logger.debug(f"Removed code block markers from stream chunk: {text}")
                
                openai_chunk["choices"][0]["delta"] = {"content": text}
                return openai_chunk
            else:
                return None  # Skip empty content

        elif event_type == "stream-end":
            # Final chunk with finish reason
            finish_reason_map = {
                "COMPLETE": "stop",
                "MAX_TOKENS": "length",
                "ERROR": "error",
                "CANCELLED": "stop",
                "SAFETY": "content_filter"
            }
            finish_reason = finish_reason_map.get(cohere_chunk.get("finish_reason"), "stop")
            openai_chunk["choices"][0]["delta"] = {}
            openai_chunk["choices"][0]["finish_reason"] = finish_reason

            # Add usage if available
            if "response" in cohere_chunk and "usage" in cohere_chunk["response"]:
                usage = cohere_chunk["response"]["usage"]
                if usage:
                    openai_chunk["usage"] = {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    }

            return openai_chunk

        # Handle other Cohere formats
        elif "is_finished" in cohere_chunk:
            if cohere_chunk.get("is_finished", False):
                # This is the final chunk
                finish_reason = "stop"
                openai_chunk["choices"][0]["delta"] = {}
                openai_chunk["choices"][0]["finish_reason"] = finish_reason
                return openai_chunk
            else:
                # This is a content chunk
                text = cohere_chunk.get("text", "")
                if text:
                    # Filter code block markers
                    if "```" in text:
                        text = text.replace("```json", "").replace("```", "")
                    
                    openai_chunk["choices"][0]["delta"] = {"content": text}
                    return openai_chunk
                else:
                    return None

        # Handle generation chunk format
        elif "generation" in cohere_chunk:
            generation = cohere_chunk.get("generation", "")
            if generation:
                # Filter code block markers
                if "```" in generation:
                    generation = generation.replace("```json", "").replace("```", "")
                
                openai_chunk["choices"][0]["delta"] = {"content": generation}
                return openai_chunk
            else:
                return None

        # Handle Cohere's streaming response format
        elif chunk_type == "text":
            # Extract text from the text field
            text = cohere_chunk.get("text", "")
            if text:
                # Filter code block markers
                if "```" in text:
                    text = text.replace("```json", "").replace("```", "")
                
                openai_chunk["choices"][0]["delta"] = {"content": text}
                return openai_chunk
            else:
                return None

        # Try to extract text from any field that might contain it
        else:
            # Look for any field that might contain text content
            for key, value in cohere_chunk.items():
                if isinstance(value, str) and len(value) > 0 and key not in ["event_type", "finish_reason", "type"]:
                    logger.debug(f"Found potential text content in field '{key}': {value}")
                    
                    # Filter code block markers if present
                    if "```" in value:
                        value = value.replace("```json", "").replace("```", "")
                    
                    openai_chunk["choices"][0]["delta"] = {"content": value}
                    return openai_chunk

            # If we get here, we couldn't find any text content
            logger.warning(f"Unknown Cohere stream chunk format: {cohere_chunk}")
            return None

    except Exception as e:
        logger.exception(f"Error transforming Cohere stream chunk: {e}. Chunk: {cohere_chunk}")
        return None


# --- Image Generation Transformation Functions ---

def transform_openai_request_to_ideogram(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms an OpenAI Image Generation request to an Ideogram API request."""
    logger.debug("Transforming OpenAI request to Ideogram format.")

    # Create the inner request object
    inner_request = {
        "prompt": openai_data.get("prompt", ""),
        "aspect_ratio": "ASPECT_1_1",  # Default to square using correct format
        "style": "natural",  # Default style
    }

    # Map size to aspect_ratio using the correct format expected by Ideogram API
    size = openai_data.get("size", "1024x1024")
    if size == "1024x1024":
        inner_request["aspect_ratio"] = "ASPECT_1_1"
    elif size == "1792x1024":
        inner_request["aspect_ratio"] = "ASPECT_16_9"
    elif size == "1024x1792":
        inner_request["aspect_ratio"] = "ASPECT_9_16"
    elif size == "512x512":
        inner_request["aspect_ratio"] = "ASPECT_1_1"
        # Remove width and height as they might not be needed with aspect_ratio
        # and could potentially conflict

    # Map style if provided
    if "style" in openai_data:
        openai_style = openai_data["style"]
        if openai_style == "vivid":
            inner_request["style"] = "enhance"
        elif openai_style == "natural":
            inner_request["style"] = "natural"

    # Map number of images
    if "n" in openai_data:
        inner_request["n"] = openai_data["n"]

    # Map response format
    if openai_data.get("response_format") == "b64_json":
        inner_request["response_format"] = "b64_json"

    # Wrap the request in the required 'image_request' property
    ideogram_request = {
        "image_request": inner_request
    }

    logger.debug(f"Transformed Ideogram request: {ideogram_request}")
    return ideogram_request


def transform_ideogram_response_to_openai(ideogram_data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Transforms an Ideogram API response to OpenAI Image Generation format."""
    logger.debug(f"Transforming Ideogram response for model {model_id}")

    openai_response = {
        "created": int(time.time()),
        "data": []
    }

    # The response might be nested under 'response' key
    response_obj = ideogram_data.get("response", ideogram_data)

    # Extract image URLs from Ideogram response
    # Check various possible locations for the image data
    images = []

    # Try different possible paths to find the images
    if "images" in response_obj:
        images = response_obj["images"]
    elif "results" in response_obj:
        images = response_obj["results"]
    elif "data" in response_obj and isinstance(response_obj["data"], list):
        images = response_obj["data"]
    # If response contains a direct image object
    elif "url" in response_obj or "b64_json" in response_obj:
        images = [response_obj]

    logger.debug(f"Found images data: {images}")

    if isinstance(images, list):
        for image in images:
            image_data = {}

            # Handle different possible response formats
            if isinstance(image, str):
                # If image is directly a URL string
                image_data["url"] = image
            elif isinstance(image, dict):
                # If image is an object with URL or base64 data
                if "url" in image:
                    image_data["url"] = image["url"]
                elif "image_url" in image:
                    image_data["url"] = image["image_url"]
                elif "path" in image:
                    image_data["url"] = image["path"]
                elif "b64_json" in image:
                    image_data["b64_json"] = image["b64_json"]
                elif "base64" in image:
                    image_data["b64_json"] = image["base64"]

            if image_data:
                openai_response["data"].append(image_data)

    logger.debug(f"Transformed OpenAI response: {openai_response}")
    return openai_response


def transform_openai_request_to_recraft(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms an OpenAI Image Generation request to a Recraft API request."""
    logger.debug("Transforming OpenAI request to Recraft format.")

    recraft_request = {
        "prompt": openai_data.get("prompt", ""),
        "width": 1024,
        "height": 1024,
        "num_images": openai_data.get("n", 1),
    }

    # Map size to width and height
    size = openai_data.get("size", "1024x1024")
    if "x" in size:
        try:
            width, height = map(int, size.split("x"))
            recraft_request["width"] = width
            recraft_request["height"] = height
        except ValueError:
            logger.warning(f"Invalid size format: {size}, using default 1024x1024")

    # Map quality if provided
    if "quality" in openai_data:
        quality = openai_data["quality"]
        if quality == "hd":
            recraft_request["quality"] = "high"
        else:
            recraft_request["quality"] = "standard"

    logger.debug(f"Transformed Recraft request: {recraft_request}")
    return recraft_request


def transform_recraft_response_to_openai(recraft_data: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Transforms a Recraft API response to OpenAI Image Generation format."""
    logger.debug(f"Transforming Recraft response for model {model_id}")

    # Use the created timestamp from the response if available
    created = recraft_data.get("response", {}).get("created", int(time.time()))

    openai_response = {
        "created": created,
        "data": []
    }

    # Extract image data from Recraft response
    # The actual response is nested under 'response.data'
    response_obj = recraft_data.get("response", {})
    images = response_obj.get("data", [])

    if isinstance(images, list):
        for image in images:
            image_data = {}

            # Handle different possible response formats
            if isinstance(image, str):
                # If image is directly a URL string
                image_data["url"] = image
            elif isinstance(image, dict):
                # If image is an object with URL or base64 data
                if "url" in image:
                    image_data["url"] = image["url"]
                elif "image_id" in image and "url" in image:
                    # Recraft specific format with image_id and url
                    image_data["url"] = image["url"]
                elif "base64" in image:
                    image_data["b64_json"] = image["base64"]
                elif "b64_json" in image:
                    image_data["b64_json"] = image["b64_json"]

            if image_data:
                openai_response["data"].append(image_data)

    # If response_format was 'b64_json', ensure we return base64 data
    # This would require additional handling if Recraft doesn't directly provide base64

    logger.debug(f"Transformed OpenAI response: {openai_response}")
    return openai_response


if __name__ == "__main__":
    import uvicorn
    # For local development, run directly:
    # Set MODEL_LIST_URL env var or create .env file
    # Example: export MODEL_LIST_URL='http://example.com/models.json'
    # Or create a .env file with: MODEL_LIST_URL=http://example.com/models.json
    uvicorn.run(app, host="127.0.0.1", port=8000)
