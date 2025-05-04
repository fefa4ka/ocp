# OpenAI Compatible API Proxy

## Overview

This project provides a proxy server that exposes OpenAI-compatible API endpoints (`/v1/models` and `/v1/chat/completions`). It dynamically fetches a list of available models from a configured source URL and routes incoming chat completion requests to the appropriate backend service based on the requested model.

This allows users to interact with various underlying LLM providers (like OpenAI, Fireworks, Anthropic, etc.) through a single, unified OpenAI-compatible interface.

## Features

*   **OpenAI Compatibility:** Exposes standard `/v1/models` and `/v1/chat/completions` endpoints.
*   **Dynamic Model Loading:** Fetches the list of available models and their backend endpoints from a remote URL at startup or periodically.
*   **Request Proxying:** Forwards `/v1/chat/completions` requests to the correct backend API specified by the `handle` in the model list.
*   **Centralized Access:** Provides a single point of access for multiple LLM backends.

## How it Works

1.  **Model List Fetching:** The proxy server starts by fetching a JSON list of available models from a specified source URL. This list contains details for each model, including its identifier (`model_version`), its family (`model_family`), and the specific backend API path (`handle`) to use for generation.
2.  **/v1/models Endpoint:** When a request is made to `/v1/models`, the proxy transforms the fetched model list into the standard OpenAI models list format and returns it. The `model_version` is typically used as the model `id`.
3.  **/v1/chat/completions Endpoint:**
    *   A client sends a request to `/v1/chat/completions`, specifying a `model` in the request body.
    *   The proxy looks up the requested `model` (matching `model_version`) in its cached model list.
    *   It retrieves the corresponding `handle` (e.g., `/openai/v1/chat/completions`, `/fireworks/chat/completions`, `/anthropic/v1/messages`).
    *   The proxy constructs the full backend URL (likely combining a base URL for the provider with the `handle`).
    *   It forwards the original request payload (potentially transforming it if the backend API signature differs significantly, although many chat APIs are similar).
    *   It receives the response from the backend service.
    *   It transforms the response back into the standard OpenAI chat completion format if necessary.
    *   It returns the response to the client.

## Configuration

The proxy requires configuration, primarily the URL for the model list and potentially an authentication token. These can be set via environment variables or by creating a `.env` file in the project root.

*   **`MODEL_LIST_URL`**: The URL to fetch the JSON list of models (e.g., `https://api.example.com/models`). Defaults to `https://api.eliza.yandex.net/models` if not set.
*   **`MODEL_LIST_AUTH_TOKEN`**: (Optional) An OAuth token to include in the `Authorization: OAuth <token>` header when fetching the model list. If not provided, the request will be made without an Authorization header.
*   **`LOG_LEVEL`**: (Optional) Sets the logging level for the application. Accepts standard Python logging levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Also accepts `NONE` to disable logging. Defaults to `INFO`.

## Setup and Running

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the Model List URL:**
    *   **Option A: Environment Variable:**
        export MODEL_LIST_URL="<your_actual_model_list_source_url>"
        export MODEL_LIST_AUTH_TOKEN="<your_oauth_token>" # Optional
        export LOG_LEVEL="DEBUG" # Optional, defaults to INFO
        ```
    *   **Option B: `.env` file:** Create a file named `.env` in the project root with the following content:
        ```dotenv
        MODEL_LIST_URL=<your_actual_model_list_source_url>
        MODEL_LIST_AUTH_TOKEN=<your_oauth_token> # Optional
        LOG_LEVEL=DEBUG # Optional, defaults to INFO
        ```

4.  **Run the server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *   `--reload` enables auto-reloading during development. Remove it for production.

5.  **Access the API:**
    *   Models list: `http://localhost:8000/v1/models`
    *   API Docs (Swagger UI): `http://localhost:8000/docs`
    *   Health Check: `http://localhost:8000/health`
    *   Chat Completions: `http://localhost:8000/v1/chat/completions` (POST)

## API Endpoints

### `GET /v1/models`

Returns a list of models available through the proxy, conforming to the OpenAI API specification.

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o-audio-preview-2024-10-01",
      "object": "model",
      "created": 1686935000, // Example timestamp
      "owned_by": "openai" // Or inferred from handle/family
    },
    {
      "id": "accounts/fireworks/models/llama-v3p1-405b-instruct",
      "object": "model",
      "created": 1686935000, // Example timestamp
      "owned_by": "fireworks" // Or inferred from handle/family
    },
    {
      "id": "claude-3-7-sonnet-20250219",
      "object": "model",
      "created": 1686935000, // Example timestamp
      "owned_by": "anthropic" // Or inferred from handle/family
    }
    // ... other models
  ]
}
```

### `POST /v1/chat/completions`

Accepts standard OpenAI chat completion requests. It looks up the requested `model` in the list fetched from `MODEL_LIST_URL`, finds its associated `handle`, and proxies the request to the backend service.

**Proxy Behavior:**
*   The target backend URL is constructed using the **scheme and host** from the `MODEL_LIST_URL` combined with the `handle` specified for the model in the source list.
    *   Example: If `MODEL_LIST_URL` is `https://api.example.com/models` and the handle for model `GigaChat-2-Max` is `/gigachat/api/v1/chat/completions`, the request will be proxied to `https://api.example.com/gigachat/api/v1/chat/completions`.
*   The request to the backend uses the **same `Authorization: OAuth <token>` header** as used for fetching the model list, if `MODEL_LIST_AUTH_TOKEN` is configured. If the token is not configured, the request is sent without an `Authorization` header.
*   The request body sent by the client is forwarded directly to the backend.
*   The response body and status code from the backend are returned directly to the client.

**Example Request:**

```json
{
  "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "temperature": 0.7
}
```

Assuming `MODEL_LIST_URL` is `https://api.example.com/models` and `MODEL_LIST_AUTH_TOKEN` is set, the proxy will:
1. Find the model `accounts/fireworks/models/llama-v3p1-405b-instruct`.
2. Get its handle (e.g., `/fireworks/chat/completions`).
3. Send a POST request to `https://api.example.com/fireworks/chat/completions`.
4. Include the `Authorization: OAuth <token>` header.
5. Forward the JSON body above.
6. Return the response received from `https://api.example.com/...` to the client.

## Example Source Model List Format

The proxy expects the model list fetched from `MODEL_LIST_URL` to be a JSON object with a `models` key containing an array of objects, similar to this:

```json
{
  "models": [
    {
      "model_version": "gpt-4o-audio-preview-2024-10-01",
      "model_family": "gpt-4",
      "handle": "/openai/v1/chat/completions",
      "prompt_price_1k": 0.0025,
      "completion_price_1k": 0.01
    },
    {
      "model_version": "accounts/fireworks/models/llama-v3p1-405b-instruct",
      "model_family": "fireworks",
      "handle": "/fireworks/chat/completions",
      "prompt_price_1k": 0.003,
      "completion_price_1k": 0.003
    },
    {
      "model_version": "claude-3-7-sonnet-20250219",
      "model_family": "claude-3",
      "handle": "/anthropic/v1/messages",
      "prompt_price_1k": 0.003,
      "completion_price_1k": 0.015
    }
    // ... other models
  ]
}
  {
    "model_version": "gpt-4o-audio-preview-2024-10-01",
    "model_family": "gpt-4",
    "handle": "/openai/v1/chat/completions",
    "prompt_price_1k": 0.0025,
    "completion_price_1k": 0.01
  },
  {
    "model_version": "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "model_family": "fireworks",
    "handle": "/fireworks/chat/completions",
    "prompt_price_1k": 0.003,
    "completion_price_1k": 0.003
  },
  {
    "model_version": "claude-3-7-sonnet-20250219",
    "model_family": "claude-3",
    "handle": "/anthropic/v1/messages",
    "prompt_price_1k": 0.003,
    "completion_price_1k": 0.015
  }
```

## Future Enhancements

*   Authentication handling (passing API keys to backends).
*   Request/Response transformation for non-compatible backends.
*   Authentication handling (passing API keys to backends, currently only supports forwarding the MODEL_LIST_AUTH_TOKEN).
*   Request/Response transformation for non-compatible backends (currently supports OpenAI, Anthropic, Gemini, and Cohere).
*   More sophisticated routing strategies (e.g., based on model family AND handle, allowing different base URLs for different providers even if served from the same model list endpoint).
*   Load balancing.
*   Support for other OpenAI endpoints (e.g., Embeddings).
*   Detailed logging and monitoring.
*   Periodic refresh of the model list (currently only fetches on startup).

### `POST /v1/images/generations`

Accepts standard OpenAI image generation requests. It looks up the requested `model` in the list fetched from `MODEL_LIST_URL`, finds its associated `handle`, and proxies the request to the backend service.

**Supported Model Families:**
* `dall-e-3`: OpenAI's DALL-E 3 image generation models
* `ideogram`: Ideogram AI image generation models
* `recraft`: Recraft AI image generation models
* `cohere`: Cohere's Command and Command-R models

**Example Request:**

```json
{
  "model": "dall-e-3",
  "prompt": "A cute baby sea otter",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard",
  "style": "vivid"
}
```

**Example Response:**

```json
{
  "created": 1684161139,
  "data": [
    {
      "url": "https://example.com/images/generated-image.png"
    }
  ]
}
```

The proxy handles the necessary transformations between the OpenAI API format and the specific formats required by different image generation backends.
