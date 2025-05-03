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

The proxy requires configuration, primarily the URL from which to fetch the model list. This might be set via environment variables or a configuration file.

*   `MODEL_LIST_URL`: The URL to fetch the JSON list of models.

## Usage

_(Instructions on how to build and run the proxy server will go here once implemented. e.g., using Docker, Python Flask/FastAPI, Node.js Express, etc.)_

```bash
# Example: How to run the server (replace with actual command)
export MODEL_LIST_URL="<your_model_list_source_url>"
python proxy_server.py
```

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

Accepts standard OpenAI chat completion requests and proxies them to the appropriate backend based on the `model` field in the request body.

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

The proxy will identify the `handle` for this model (`/fireworks/chat/completions`) and forward the request accordingly. The response will be formatted like a standard OpenAI chat completion response.

## Example Source Model List Format

The proxy expects the model list fetched from `MODEL_LIST_URL` to be a JSON array with objects similar to this:

```json
[
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
```

## Future Enhancements

*   Authentication handling (passing API keys to backends).
*   Request/Response transformation for non-compatible backends.
*   Load balancing or routing strategies.
*   Caching of model list.
*   Support for other OpenAI endpoints (e.g., Embeddings).
*   Detailed logging and monitoring.
