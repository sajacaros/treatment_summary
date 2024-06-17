import os
from pathlib import Path

import dotenv
import nlptutti as metrics

dotenv.load_dotenv()
data_path = os.environ.get('DATA_PATH')


def evaluate(filepath: str, ref_dir: str = 'answer', verbose=False) -> tuple[float, float]:
    """
    STT 결과를 평가(CER, WER)하는 함수
    예측과 정답의 파일명은 동일해야 함

    Args:
        filepath(str): 예측 파일의 path
        ref_dir(str): 정답지 디렉토리
        verbose(bool): default: False
    Returns:
         cer(float): CER 결과
         wer(float): WER 결과
    """
    with open(os.path.join(data_path, ref_dir, Path(filepath).name), encoding='utf-8') as f:
        answer_text = f.read()
    with open(os.path.join(data_path, filepath), encoding='utf-8') as f:
        preds_text = f.read()
    cer_result = metrics.get_cer(answer_text, preds_text)
    wer_result = metrics.get_wer(answer_text, preds_text)
    if verbose:
        print(f"{Path(filepath).stem}, cer: {cer_result['cer']:.4f}, wer: {wer_result['wer']:.4f}")

    return cer_result['cer'], wer_result['wer']
