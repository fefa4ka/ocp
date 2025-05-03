import time
from typing import List, Optional

from pydantic import BaseModel, Field


# --- Source Model Definition ---
class SourceModel(BaseModel):
    """Represents the structure of a model object from the source list."""

    model_version: str = Field(..., description="Unique identifier for the model version.")
    model_family: str = Field(..., description="Family the model belongs to (e.g., gpt-4, llama).")
    handle: str = Field(..., description="API path handle for the generation endpoint.")
    prompt_price_1k: Optional[float] = Field(None, description="Price per 1k prompt tokens.")
    completion_price_1k: Optional[float] = Field(None, description="Price per 1k completion tokens.")


class SourceModelList(BaseModel):
    """Represents the top-level structure of the source model list JSON."""
    models: List[SourceModel]


# --- OpenAI Model Definition ---
class OpenAIModel(BaseModel):
    """Represents the structure of a model object in OpenAI's /v1/models format."""

    id: str = Field(..., description="The model identifier, which can be referenced in the API endpoints.")
    object: str = Field("model", description="The object type, which is always 'model'.")
    created: int = Field(default_factory=lambda: int(time.time()), description="The Unix timestamp (in seconds) when the model was created.")
    owned_by: str = Field("proxy", description="The organization that owns the model (inferred or fixed).")


class OpenAIModelList(BaseModel):
    """Represents the structure of the list returned by OpenAI's /v1/models endpoint."""

    object: str = Field("list", description="The object type, which is always 'list'.")
    data: List[OpenAIModel] = Field(..., description="A list of model objects.")
