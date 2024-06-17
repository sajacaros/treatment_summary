import os
import sys
from pathlib import Path

import dotenv
import numpy as np

from dukim.evaluation import stt_evaluation
from dukim.stt.engine import WhisperLocalSTT, stt, WhisperSTT
from dukim.transform.audio_pipe import preprocessing_vad, preprocessing_simple, preprocessing_vad_split, \
    preprocessing_split

dotenv.load_dotenv()
data_path = os.environ.get('DATA_PATH')


def print_statistics(cer_list, wer_list):
    print('-' * 20)
    print('         mean     std')
    print(f'cer: {np.mean(cer_list):.4f}, {np.std(cer_list):.4f}')
    print(f'wer: {np.mean(wer_list):.4f}, {np.std(wer_list):.4f}')


def stt_worker(is_owner=False, transformers=None, evaluation=False, adjust_temperature=True, max_try=5, max_temperature=0.4):
    if is_owner:
        stt_engine = WhisperSTT(api_key=os.environ.get('OPENAI_API_KEY'), transformers=transformers)
    else:
        stt_engine = WhisperLocalSTT(transformers=transformers)
    allow_extensions = ('.m4a', '.mp3', '.wav')
    filelist, has_repeated_word_list = [], []
    for audio_filename in os.listdir(os.path.join(data_path, 'origin')):
        if len(audio_filename.split('.')) > 1 and audio_filename.lower().endswith(allow_extensions):
            filepath, has_repeated_word, _, _ = stt(stt_engine, audio_filename, adjust_temperature=adjust_temperature, max_try=max_try, max_temperature=max_temperature)
            filelist.append(filepath)
            has_repeated_word_list.append(has_repeated_word)
    if evaluation:
        cer_list, wer_list = [], []
        for text_filepath, has_repeated_word in zip(filelist, has_repeated_word_list):
            cer, wer = stt_evaluation.evaluate(text_filepath, ref_dir='answer', verbose=False)
            print(f"{'* ' if has_repeated_word else ''}{Path(text_filepath).stem}, cer: {cer:.4f}, wer: {wer:.4f}")
            if not has_repeated_word:
                cer_list.append(cer)
                wer_list.append(wer)
        print_statistics(cer_list, wer_list)


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    # stt_worker(is_owner=True, transformers=preprocessing_split(), evaluation=True, adjust_temperature=False)
    stt_worker(is_owner=False, transformers=preprocessing_vad(), evaluation=True)
    # stt_worker(is_owner=False, transformers=preprocessing_simple(), evaluation=True)
    # stt_worker(is_owner=True, transformers=preprocessing_vad_split(), evaluation=True)
