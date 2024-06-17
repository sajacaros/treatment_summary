from typing import Optional

from pydantic import BaseModel


class ChartingRequest(BaseModel):
    text_path: str
    llm: str
    prompt: str


class LLMResource(BaseModel):
    total_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_cost: Optional[str]


class ChartingResponse(BaseModel):
    charting_filepath: str
    resource: LLMResource
    progress_time: float