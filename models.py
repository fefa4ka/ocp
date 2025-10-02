import time
from typing import List, Optional, Dict, Any, Union, Literal

from pydantic import BaseModel, Field, HttpUrl


# --- Source Model Definition ---
class SourceModel(BaseModel):
    """Represents the structure of a model object from the source list."""

    model_version: str = Field(..., description="Unique identifier for the model version.")
    model_family: str = Field(..., description="Family the model belongs to (e.g., gpt-4, llama).")
    handle: str = Field(..., description="API path handle for the generation endpoint.")
    prompt_price_1k: Optional[float] = Field(None, description="Price per 1k prompt tokens.")
    completion_price_1k: Optional[float] = Field(None, description="Price per 1k completion tokens.")
    prices: Optional[Dict[str, Union[str, float]]] = Field(None, description="Pricing information for different metrics.")
    discription: Optional[str] = Field(None, description="Description of the model.")
    specific_versions: Optional[Any] = Field(None, description="Specific version information.")


class SourceModelList(BaseModel):
    """Represents the top-level structure of the source model list JSON."""
    models: List[SourceModel]


# --- OpenAI Model Definition ---
class OpenAIModel(BaseModel):
    """Represents the structure of a model object in OpenAI's /v1/models format."""

    id: str = Field(..., description="The model identifier, which can be referenced in the API endpoints.")
    object: str = Field(default="model", description="The object type, which is always 'model'.")
    created: int = Field(default_factory=lambda: int(time.time()), description="The Unix timestamp (in seconds) when the model was created.")
    owned_by: str = Field(default="proxy", description="The organization that owns the model (inferred or fixed).")


class OpenAIModelList(BaseModel):
    """Represents the structure of the list returned by OpenAI's /v1/models endpoint."""

    object: str = Field("list", description="The object type, which is always 'list'.")
    data: List[OpenAIModel] = Field(..., description="A list of model objects.")


# --- OpenAI Chat Completion Definitions ---

# Using Any for messages and choices for flexibility, can be refined later
class OpenAIChatCompletionRequest(BaseModel):
    """Represents the request body for OpenAI's /v1/chat/completions."""
    model: str = Field(..., description="ID of the model to use (maps to model_version).")
    messages: List[Dict[str, Any]] = Field(..., description="A list of messages comprising the conversation so far.")
    # Add other common OpenAI parameters as needed, mirroring the target API if possible
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None  # Legacy parameter
    function_call: Optional[Union[str, Dict[str, Any]]] = None  # Legacy parameter
    # Include other potential fields based on OpenAI spec or backend needs


class OpenAIChatCompletionResponse(BaseModel):
     """Represents the response body for OpenAI's /v1/chat/completions."""
     # Basic structure, assuming the backend returns a compatible format.
     # If not, transformation logic will be needed in the endpoint.
     id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}") # Example ID
     object: str = "chat.completion"
     created: int = Field(default_factory=lambda: int(time.time()))
     model: str # Should match the requested model
     choices: List[Any] # Define a Choice model later if needed
     usage: Optional[Any] = None # Define a Usage model later if needed
     # Add other fields like system_fingerprint if needed


# --- OpenAI Image Generation Definitions ---

class OpenAIImageGenerationRequest(BaseModel):
    """Represents the request body for OpenAI's /v1/images/generations."""
    model: Optional[str] = Field(None, description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    n: Optional[int] = Field(1, description="The number of images to generate.")
    size: Optional[str] = Field("1024x1024", description="The size of the generated images.")
    quality: Optional[str] = Field("standard", description="The quality of the image generation.")
    response_format: Optional[str] = Field("url", description="The format in which the generated images are returned.")
    style: Optional[str] = Field("vivid", description="The style of the generated images.")
    user: Optional[str] = Field(None, description="A unique identifier for the end-user.")


class OpenAIImageData(BaseModel):
    """Represents a single image in the OpenAI image generation response."""
    url: Optional[HttpUrl] = Field(None, description="The URL of the generated image.")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image.")
    revised_prompt: Optional[str] = Field(None, description="The prompt that was used to generate the image, potentially revised from the original.")


class OpenAIImageGenerationResponse(BaseModel):
    """Represents the response body for OpenAI's /v1/images/generations."""
    created: int = Field(default_factory=lambda: int(time.time()), description="The timestamp for when the image was created.")
    data: List[OpenAIImageData] = Field(..., description="The list of generated images.")


# --- OpenAI Text-to-Speech Definitions ---

class OpenAITTSRequest(BaseModel):
    """Represents the request body for OpenAI's /v1/audio/speech."""
    model: str = Field(..., description="One of the available TTS models: tts-1 or tts-1-hd.")
    input: str = Field(..., description="The text to generate audio for. The maximum length is 4096 characters.")
    voice: str = Field(..., description="The voice to use when generating the audio. Supported voices are alloy, echo, fable, onyx, nova, and shimmer.")
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field("mp3", description="The format to audio in.")
    speed: Optional[float] = Field(1.0, description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.")
    user: Optional[str] = Field(None, description="A unique identifier for the end-user.")


class OpenAITranscriptionRequest(BaseModel):
    """Represents the request body for OpenAI's /v1/audio/transcriptions."""
    file: str = Field(..., description="The audio file object (not file name) to transcribe, in one of these formats: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, or webm.")
    model: str = Field(..., description="ID of the model to use. Only whisper-1 is currently available.")
    language: Optional[str] = Field(None, description="The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.")
    prompt: Optional[str] = Field(None, description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.")
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Field("json", description="The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.")
    temperature: Optional[float] = Field(0.0, description="The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(None, description="The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use timestamp granularities.")
    user: Optional[str] = Field(None, description="A unique identifier for the end-user.")
