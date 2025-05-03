import json # Add json import
import logging
import time # Add time import
from typing import Any, AsyncGenerator, Dict, List, Optional # Import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from models import (
    OpenAIChatCompletionRequest,  # Added import
    OpenAIModel,
    OpenAIModelList,
    SourceModel,
    SourceModelList,
)

# Set logging level to DEBUG to see the new log message
logging.basicConfig(level=logging.DEBUG)
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
        logger.debug(f"Received raw request data for chat completion: {request_data}") # Log the received data
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


        # --- API Specific Adjustments (Example for Anthropic) ---
        # Modify request_data if needed based on the target backend
        if "/anthropic/" in handle.lower():
            if "max_tokens" not in request_data or request_data.get("max_tokens") is None:
                default_max_tokens = 4096 # Set a reasonable default for Anthropic
                logger.warning(f"Anthropic request for model '{model_id}' missing 'max_tokens'. Applying default: {default_max_tokens}")
                request_data["max_tokens"] = default_max_tokens
            # Add other Anthropic-specific transformations here if needed
            # e.g., mapping OpenAI 'messages' to Anthropic 'messages' + 'system' prompt
            logger.debug(f"Anthropic request data after adjustments for model '{model_id}': {request_data}")

        # --- Forward the request ---
        is_streaming = request_data.get("stream", False)
        logger.info(f"Forwarding request for model '{model_id}' to {target_url}. Streaming: {is_streaming}")

        # Define the stream generator here, accepting necessary parameters
        async def stream_generator(
            req_url: str,
            req_data: dict,
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
            openai_chunk_id = f"chatcmpl-anthropic-{int(time.time())}" # Generate a base ID for the stream

            async with httpx.AsyncClient() as stream_client:
                try:
                    async with stream_client.stream(
                        "POST", req_url, json=req_data, headers=req_headers, timeout=180.0
                    ) as backend_response:
                        # Check for backend errors *before* streaming body
                        if backend_response.status_code >= 400:
                            error_body = await backend_response.aread()
                            error_detail = error_body.decode() or f"Backend error {backend_response.status_code}"
                            logger.error(f"Backend streaming request failed with status {backend_response.status_code}: {error_detail}")
                            # Yield an OpenAI-like error chunk before raising? Or just raise? Let's just raise.
                            raise HTTPException(status_code=backend_response.status_code, detail=error_detail)

                        logger.info(f"Starting stream processing for model '{requested_model}'. Anthropic transformation: {is_anthropic}")

                        if is_anthropic:
                            # Process Anthropic SSE stream line by line
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

                        else: # Not Anthropic, assume OpenAI compatible stream (forward raw chunks)
                            # Note: This assumes the non-Anthropic backend ALREADY sends OpenAI formatted SSE.
                            # If not, more transformation logic would be needed here too.
                            async for chunk in backend_response.aiter_bytes():
                                # We need to yield SSE formatted strings, not raw bytes
                                # Assuming the chunk itself is the 'data' part of an SSE message
                                yield f"data: {chunk.decode()}\n\n" # Decode bytes and format as SSE
                            logger.info(f"Finished forwarding raw stream from backend for model '{requested_model}'")

                        # Send the final [DONE] message for all streams
                        yield "data: [DONE]\n\n"
                        logger.info(f"Sent [DONE] message for model '{requested_model}'.")

                except HTTPException:
                     raise # Re-raise HTTP exceptions from status check
                except Exception as e:
                    # Log errors occurring during the streaming process itself
                    logger.exception(f"Error during backend stream processing for model {requested_model}: {e}")
                    # Yielding an error message might be better for the client than just stopping.
                    error_payload = {
                        "error": {
                            "message": f"Proxy error during streaming: {e}",
                            "type": "proxy_error",
                            "code": "stream_processing_error"
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n" # Still send DONE after error? OpenAI spec implies yes.


        # Now, handle the request based on streaming flag
        try:
            if is_streaming:
                # Create the generator instance, passing the required arguments
                generator = stream_generator(target_url, request_data, headers_to_forward, handle, model_id)
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
                    logger.debug(f"Backend response data for model '{model_id}': {response_data}")

                    # --- Transform response if needed ---
                    final_response_data = response_data
                    if "/anthropic/" in handle.lower():
                        logger.info(f"Transforming Anthropic response for model '{model_id}' to OpenAI format.")
                        final_response_data = transform_anthropic_response_to_openai(response_data, model_id)
                        # Check if transformation resulted in an error structure
                        if "error" in final_response_data:
                             # Use a client error status code if transformation failed badly
                             # Or maybe 502 Bad Gateway if backend response was unusable? Let's use 500 for now.
                             logger.error(f"Transformation failed, returning error response: {final_response_data}")
                             return JSONResponse(content=final_response_data, status_code=500)

                    # Return potentially transformed data with original success status code
                    return JSONResponse(content=final_response_data, status_code=backend_response.status_code)

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
                    "type": "proxy_error",
                    "code": "invalid_backend_response"
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
        logger.exception(f"Error transforming Anthropic response: {e}. Original data: {anthropic_data}")
        # Return a generic error structure
        return {
            "error": {
                "message": f"Error transforming response from Anthropic backend: {e}",
                "type": "proxy_error",
                "code": "transformation_error"
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
        logger.exception(f"Error processing Anthropic event type {event_type} data {event_data}: {e}")
        return None # Skip chunk on error


if __name__ == "__main__":
    import uvicorn
    # For local development, run directly:
    # Set MODEL_LIST_URL env var or create .env file
    # Example: export MODEL_LIST_URL='http://127.0.0.1:8001/models.json'
    # Or create a .env file with: MODEL_LIST_URL=http://127.0.0.1:8001/models.json
    uvicorn.run(app, host="0.0.0.0", port=8000)
