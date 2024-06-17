from pydantic import BaseModel


class STSRequest(BaseModel):
    method: str
    filename: str
    upload_dir: str


class STSResponse(BaseModel):
    text_path: str
    retry: int
    temperature: float
    progress_time: float