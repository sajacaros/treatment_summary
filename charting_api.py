import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import uvicorn
from fastapi import FastAPI

from dukim.model.charting_model import ChartingResponse, ChartingRequest, LLMResource
from dukim.model.sts_model import STSResponse, STSRequest
from dukim.stt.engine import stt, WhisperLocalSTT, WhisperSTT
from dukim.summary.conversation_chat import processing_charting
from dukim.transform.audio_pipe import preprocessing_vad, preprocessing_vad_split

app = FastAPI()
dotenv.load_dotenv()
data_path = os.environ.get('DATA_PATH')
charting_dir = 'charting'
charting_path = os.path.join(data_path, charting_dir)
local_engine = WhisperLocalSTT(transformers=preprocessing_vad())
api_engine = WhisperSTT(transformers=preprocessing_vad_split())


@app.get("/sts", status_code=200)
def sts_api():
    return 'ok'


@app.post("/sts", response_model=STSResponse)
def sts_api(request: STSRequest) -> Any:
    print(f'sts api request : {request}')
    start = time.time()

    engine = local_engine if request.method == 'local' else api_engine
    filepath, has_repeated_word, count, temperature = stt(engine=engine, audio_filename=request.filename,
                                                          origin_dir=request.upload_dir)
    end = time.time()
    gc.collect()

    return STSResponse(
        text_path=filepath,
        retry=count,
        temperature=temperature,
        progress_time=round(end - start, 2)
    )


@app.post("/charting", response_model=ChartingResponse)
def charting_api(request: ChartingRequest) -> Any:
    print(f'charting api request : {request}')
    llm = 'gpt3' if request.llm == 'gpt3.5' else request.llm
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    filename = f"{Path(request.text_path).stem}_{llm}_{request.prompt[0]}_{timestamp}.txt"
    os.makedirs(charting_path, exist_ok=True)
    start = time.time()
    resource = processing_charting(
        file_path=os.path.join(data_path, request.text_path),
        oneshot_path=os.path.join(data_path, 'template/', 'data_one_short_content.json'),
        llm_name=llm,
        prompt_type=request.prompt,
        save_path=charting_path,
        save_file_name=filename
    )
    end = time.time()
    gc.collect()
    res = ChartingResponse(
        charting_filepath=os.path.join(charting_dir, filename),
        resource=LLMResource(
            total_tokens=resource.total_tokens,
            prompt_tokens=resource.prompt_tokens,
            completion_tokens=resource.completion_tokens,
            total_cost=f'$ {round(resource.self_total_cost, 4)}'
        ),
        progress_time=round(end - start, 2)
    )
    print(res)
    return res


if __name__ == '__main__':
    os.environ["PYTHONWARNINGS"] = 'always'
    uvicorn.run(app, host='0.0.0.0', port=8090)
