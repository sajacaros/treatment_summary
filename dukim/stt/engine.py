import datetime
import json
import os
import re
import time
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import Literal, Tuple, List, Union

import stable_whisper
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from stable_whisper import WhisperResult

from dukim.stt.util import trace
from dukim.transform.audio_pipe import IdentityPipe, AudioPipe

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_path = os.environ.get('DATA_PATH')
origin_path = os.path.join(data_path, 'origin')
mp3_path = os.path.join(data_path, 'mp3')
mp3_intro_path = os.path.join(data_path, 'mp3_intro')
wav_path = os.path.join(data_path, 'wav')
treatment_path = os.path.join(data_path, 'treatment')


class STTEngine(metaclass=ABCMeta):
    """
    audio에서 text를 추출하는 추상 클래스
    """

    @abstractmethod
    def speech_to_text(
            self,
            mp3_name: str,
            *,
            temperature: float = 0.0,
            origin_dir='origin'
    ) -> tuple[str, bool]:
        """
        audio file을 입력받아 text로 바꾸는 메서드

        Args:
            mp3_name: audio 파일의 이름
            temperature: 창의성을 제어하는 샘플링 수치, 0(정적) <-> 1(창의적), default value: 0
            origin_dir: audio 파일이 있는 디렉토리
        Returns:
            treatment_path: 상담파일 path
            has_repeated_word: 반복어구 발생 여부
        """
        pass


def extract_text_with_time(text, start_audio_time):
    pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]+?)(?=\n{2}|\Z)"
    matches = re.findall(pattern, text)

    result = []
    for match in matches:
        start_time = match[0]
        end_time = match[1]
        text = match[2].strip()
        splitted_start = start_time.split(":")
        splitted_end = end_time.split(":")
        start_seconds = (float(splitted_start[0].replace(",", ".")) * 3600
                         + float(splitted_start[1].replace(",", ".")) * 60
                         + float(splitted_start[2].replace(",", "."))
                         + start_audio_time)
        end_seconds = (float(splitted_end[0].replace(",", ".")) * 3600
                       + float(splitted_end[1].replace(",", ".")) * 60
                       + float(splitted_end[2].replace(",", "."))
                       + start_audio_time)
        result.append({"start": start_seconds, "end": end_seconds, "text": text})

    return result


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return round(audio.duration_seconds, 2)


class WhisperSTT(STTEngine):
    """
    Whisper api를 이용하여 audio에서 text를 추출하는 클래스
    """

    def __init__(self, api_key=None, transformers=None):
        """
        :param api_key: whisper api key
        """
        self.client = OpenAI(api_key=api_key)
        self.transformers = transformers if transformers else IdentityPipe()

    @trace
    def speech_to_text(
            self,
            audio_filename,
            *,
            temperature=0.0,
            origin_dir='origin'
    ) -> tuple[str, bool]:
        """
        audio file을 입력받아 text로 바꾸는 메서드

        Args:
            audio_filename: audio 파일의 이름
            temperature: 창의성을 제어하는 샘플링 수치, 0(정적) <-> 1(창의적), default value: 0
            origin_dir: audio 파일이 있는 디렉토리
        Returns:
            treatment_path: 상담파일 path
            has_repeated_word: 반복어구 발생 여부
        """
        print(f'--- STT 작업 시작 ---, {audio_filename}')
        filepath_list = self.transformers([os.path.join(origin_dir, audio_filename)])
        transcript_list = []
        start_audio_time = 0.0
        for filepath in filepath_list:
            print(f"{filepath}'s TEXT 추출 시작")
            start = time.time()
            with open(os.path.join(data_path, filepath), 'rb') as audio_file:
                transcript_text = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model='whisper-1',
                    language="ko",
                    prompt='You should end with period in every sentences.',
                    response_format='srt',
                    temperature=temperature
                )
            end = time.time()
            times = str(datetime.timedelta(seconds=end - start))
            print(f"{filepath}'s STT, 걸린 시간 : {times.split('.')[0]}")

            if transcript_text:
                transcript_list.extend(extract_text_with_time(transcript_text, start_audio_time))
            else:
                print(f'{filepath} STT 추출 실패')
            start_audio_time += get_audio_duration(os.path.join(data_path, filepath))
        # 반목어구 체크
        concat_text = ' '.join([t['text'] for t in transcript_list])
        if has_repeated_word := has_repeated_words_in_window(concat_text):
            print(f'repeated occurred - {has_repeated_word=}')

        if len(transcript_list) > 0:
            title = ''.join(audio_filename.split('.')[:-1])
            output_filepath = self.to_output(transcript_list, title, format='text')
            self.to_output(transcript_list, title, format='json')
        else:
            print(f'{audio_filename} - text 추출 실패')
            raise RuntimeError()
        return (output_filepath, has_repeated_word)

    @staticmethod
    def to_output(texts, title, format: Literal['text', 'json'] = 'text', verbose=False):
        os.makedirs(treatment_path, exist_ok=True)
        if format == 'text':
            filename = f'{title}.txt'
            combined_text = ' '.join([t['text'] for t in texts])
        elif format == 'json':
            filename = f'{title}.json'
            combined_text = json.dumps(texts, allow_nan=True, ensure_ascii=False)
        else:
            raise NotImplemented

        with open(os.path.join(treatment_path, filename), mode='w', encoding='utf-8') as f:
            if verbose:
                print(combined_text)
            f.write(combined_text)
        return os.path.join('treatment', filename)


def has_repeated_words_in_window(sentence, threshold=10, kernel_size=50, strid=10):
    words = re.findall(r'\b\w+\b', sentence)
    length = len(words)

    for start in range(0, length - kernel_size + 1, strid):
        end = start + kernel_size
        window = ' '.join(words[start:end])
        word_counts = Counter(window.split())
        exclude_words = ['네']
        repeated_words = [word for word, count in word_counts.items() if
                          count > threshold and word not in exclude_words]

        if repeated_words:
            print(f"repeated_words : [{', '.join(repeated_words) if repeated_words else ''}]")
            return True

    return False


def whisper_result_to_json(whisper_result):
    return [{'start': segment.start, 'end': segment.end, 'text': segment.text} for segment in whisper_result]


class WhisperLocalSTT(STTEngine):
    """
    local에 whisper model을 다운로드하여 audio에서 text를 추출하는 클래스

    Args:
        size: model size, default : medium
            base	74 M	~1 GB
            small	244 M	~2 GB
            medium	769 M	~5 GB
            large	1550 M	~10 GB
        retryable: 변환 실패시 총 3회 반복 시도
    """

    def __init__(self, size='medium', retryable=3, transformers: AudioPipe = None):
        self.model = stable_whisper.load_model(size, device='cuda')
        self.retryable = retryable
        self.transformers = transformers if transformers else IdentityPipe()

    @trace
    def speech_to_text(
            self,
            audio_filename: str,
            *,
            title=None,
            temperature=0.0,
            origin_dir='origin'
    ) -> Tuple[str, bool]:
        """
        audio file을 입력받아 text로 바꾸는 메서드

        Args:
            audio_filename: audio 파일의 이름
            temperature: 창의성을 제어하는 샘플링 수치, 0(정적) <-> 1(창의적), default value: 0
            origin_dir: audio 파일이 있는 디렉토리
        Returns:
            treatment_path: 상담파일 path
            has_repeated_word: 반복어구 발생 여부
        """
        print(f'--- STT 작업 시작 ---, {audio_filename}')
        audio_filepath = os.path.join(origin_dir, audio_filename)
        filepath_list = self.transformers([audio_filepath])
        texts = self.extract_and_collect(filepath_list, temperature=temperature)

        # 반목어구 체크
        if has_repeated_word := has_repeated_words_in_window(''.join([t.text for t in texts])):
            print(f'repeated occurred - {has_repeated_word=}')

        if not title:
            title = ''.join(audio_filename.split('.')[:-1])
        output_filepath = self.to_output(texts, title, output_format='text')
        self.to_output(texts, title, output_format='json')
        return output_filepath, has_repeated_word

    def extract_text(self, filepath, temperature=0.0) -> tuple[bool, Union[WhisperResult, None]]:
        try:
            ret = self.model.transcribe(
                temperature=temperature,
                language='ko',
                audio=os.path.join(data_path, filepath),
                initial_prompt='You should end with period in every sentences.',
                verbose=False,
            )
        except Exception as e:
            print(f'audio({filepath}) stt error, error : ({e})')
            return False, None
        return True, ret

    @staticmethod
    def to_output(texts, title, output_format: Literal['text', 'json'] = 'text', verbose=False):
        os.makedirs(treatment_path, exist_ok=True)
        if output_format == 'text':
            filename = f'{title}.txt'
            combined_text = ''.join([t.text for t in texts])
        elif output_format == 'json':
            filename = f'{title}.json'
            json_texts = []
            for whisper_result in texts:
                json_texts.extend(whisper_result_to_json(whisper_result))
            combined_text = json.dumps(json_texts, allow_nan=True, ensure_ascii=False)
        else:
            raise NotImplemented

        with open(os.path.join(treatment_path, filename), mode='w', encoding='utf-8') as f:
            if verbose:
                print(combined_text)
            f.write(combined_text)
        return os.path.join('treatment', filename)

    def retryable_extract(self, filepath, temperature=0.0) -> Tuple[bool, Union[WhisperResult, None]]:
        ret = False
        ret_text = None
        retry_count = 0
        while not ret and retry_count < self.retryable:
            ret, ret_text = self.extract_text(filepath, temperature=temperature)
            retry_count += 1
        return ret, ret_text

    def extract_and_collect(
            self,
            filepath_list: list,
            temperature=0.0
    ) -> List[WhisperResult]:
        texts = []
        start_audio_time = 0.0
        for filepath in filepath_list:
            print(f"{filepath}'s TEXT 추출 시작")
            start = time.time()
            ret, ret_text = self.retryable_extract(filepath, temperature)
            end = time.time()
            times = str(datetime.timedelta(seconds=end - start))
            print(f"{filepath}'s STT, 걸린 시간 : {times.split('.')[0]}")

            if ret:
                for segment in ret_text:
                    segment.start = round(segment.start + start_audio_time, 2)
                    segment.end = round(segment.end + start_audio_time, 2)
                texts.append(ret_text)
            else:
                print(f'{filepath} STT 추출 실패')
            start_audio_time += get_audio_duration(os.path.join(data_path, filepath))
        return texts


def stt(
        engine: STTEngine,
        audio_filename: str,
        origin_dir: str = 'origin',
        *,
        adjust_temperature: bool = True,
        max_try: int = 5,
        max_temperature: float = 0.4
) -> (str, bool, int, float):
    """
    STT를 추출하고 반복어구 발생시 설정된 정보대로 재수행하는 함수

    Args:
        engine(STTEngine): STTEngine의 인스턴스
        audio_filename(str): 변환할 파일명
        origin_dir(str): 변환할 파일이 있는 디렉토리, default: 'origin'
        adjust_temperature(bool): 적응형 temperature 적용 여부, default: True
        max_try(int): 최대 retry 횟수, default: 5
        max_temperature(float): 최대 temperature값, default: 0.4
    Returns:
        treatment_path: 상담파일 path
        has_repeated_word: 반복어구 발생 여부
        int: retry 횟수
        temperature: STT에 사용된 최종 temperature
    """
    count = 0

    temperature = 0.0
    if adjust_temperature:
        while True:
            print(f'count : {count}, temperature : {temperature}')
            filepath, has_repeated_word = engine.speech_to_text(
                audio_filename, temperature=temperature, origin_dir=origin_dir
            )
            if not has_repeated_word or count == max_try:
                break

            count += 1
            temperature = min(max_temperature, temperature + 0.2)
        return filepath, has_repeated_word, count, temperature
    else:
        return *engine.speech_to_text(audio_filename, temperature=0.0,
                                      origin_dir=origin_dir), count, temperature
