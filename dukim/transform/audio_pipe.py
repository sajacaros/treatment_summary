import datetime
import gc
import os
import time
from abc import ABCMeta, abstractmethod
from typing import List

import torch
from dotenv import load_dotenv
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence, detect_silence

load_dotenv()
data_path = os.environ.get('DATA_PATH')


class AudioPipe(metaclass=ABCMeta):
    """
    PipeChain의 요소를 이루는 추상 클래스

    Args:
        des_dir(str or None): 결과를 저장할 디렉토리
        remove_input(bool): 입력물 삭제 여부, default: False
    """

    def __init__(self, des_dir: str = None, remove_input: bool = False):
        self.des_dir = des_dir
        self.remove_input = remove_input
        if self.des_dir:
            os.makedirs(os.path.join(data_path, des_dir), exist_ok=True)

    def __call__(self, filepaths: List[str]):
        print(f"{self.__repr__()}'s pipe start")

        start = time.time()
        ret = self.next(filepaths)
        end = time.time()
        times = str(datetime.timedelta(seconds=end - start))
        print(f"{self.__repr__()}'s pipe end, 걸린 시간 : {times.split('.')[0]}")
        if self.remove_input:
            for filepath in filepaths:
                try:
                    os.remove(os.path.join(data_path, filepath))
                except FileNotFoundError:
                    print(f'failed to remove, file({filepath}) not found.')

        return ret

    @abstractmethod
    def next(self, filepath_list: List[str]) -> List[str]:
        pass

    def __repr__(self):
        return f'pipe - {type(self).__name__}'


class Pipechain(AudioPipe):
    """
    PipeChain의 구현체, Pipe 요소를 순회하며 pipe를 실행
    """

    def __init__(self):
        super().__init__()
        self.pipes: List[AudioPipe] = []

    def next(self, filepaths: List[str]) -> List[str]:
        for pipe in self.pipes:
            filepaths = pipe(filepaths)
        return filepaths

    def add(self, pipe: AudioPipe):
        self.pipes.append(pipe)


def convert_audio_to_mp3(from_filepath, to_filepath, normalize=False, minimize=True):
    convert_audio_to_(from_filepath, to_filepath, format='mp3', normalize=normalize, minimize=minimize)


def convert_audio_to_wav(from_filepath, to_filepath, normalize=False, minimize=True):
    convert_audio_to_(from_filepath, to_filepath, format='wav', normalize=normalize, minimize=minimize)


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def convert_audio_to_(from_filepath, to_filepath, format, minimize=True, normalize=False, normal_method='basic',
                      basis_dbfs=-20.0):
    assert not (normalize and minimize)
    audio = AudioSegment.from_file(os.path.join(data_path, from_filepath))
    if normalize:
        if normal_method == 'basic':
            audio = effects.normalize(audio)
        elif normal_method == 'dBFS':
            audio = match_target_amplitude(audio, basis_dbfs)
        else:
            raise NotImplemented

    if minimize:
        audio = (audio
                 .set_frame_rate(16000)
                 .set_channels(1)
                 .set_sample_width(2)
                 )

    audio.export(os.path.join(data_path, to_filepath), format=format)
    del audio
    gc.collect()


def add_intro_ending(from_filepath, to_filepath):
    # 인트로/엔딩 파일 로드
    intro_clip = AudioSegment.from_file(os.path.join(data_path, 'intro.wav'))
    ending_clip = AudioSegment.from_file(os.path.join(data_path, 'ending.wav'))
    # 상담 파일 불러오기
    audio = AudioSegment.from_file(os.path.join(data_path, from_filepath))

    # 두 오디오 클립 합치기
    audio = intro_clip + audio + ending_clip

    # 결과 저장
    audio.export(os.path.join(data_path, to_filepath), format=to_filepath.split('.')[-1])


def get_filename(filepath):
    return os.path.split(filepath)[-1]


class IdentityPipe(AudioPipe):
    """
    자기 자신을 리턴하는 Pipe
    """

    def next(self, filepath_list: List[str]) -> List[str]:
        return filepath_list


class ToMP3(AudioPipe):
    """
    오디오를 MP3로 변환하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: mp3
        remove_input(bool): 입력물 삭제 여부, default: False
        minimize(bool): 오디오를 다운 샘플링(16000HZ, mono, 2 sample_width) 할 지 여부, default: True
    """

    def __init__(self, des_dir: str = 'mp3', remove_input: bool = False, minimize: bool = True):
        super().__init__(des_dir, remove_input)
        self.minimize = minimize

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            filename = get_filename(file_path)
            split_filename = filename.split('.')
            only_filename = '.'.join(split_filename[:-1])
            if split_filename[-1].lower() in ['m4a', 'wav', 'mp3']:
                mp3_filepath = os.path.join(self.des_dir, f'{only_filename}.mp3')
                convert_audio_to_mp3(file_path, mp3_filepath, minimize=self.minimize)
                ret.append(mp3_filepath)
            else:
                print(f'failed to transform - {file_path}, not supported extension')
                raise NotImplemented
        return ret


class ToWav(AudioPipe):
    """
    오디오를 Wav로 변환하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: wav
        remove_input(bool): 입력물 삭제 여부, default: False
        minimize(bool): 오디오를 다운 샘플링(16000HZ, mono, 2 sample_width) 할 지 여부, default: True
    """

    def __init__(self, des_dir: str = 'wav', remove_input: bool = False, minimize: bool = False):
        super().__init__(des_dir, remove_input)
        self.minimize = minimize

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            filename = get_filename(file_path)
            split_filename = filename.split('.')
            only_filename = '.'.join(split_filename[:-1])

            if split_filename[-1].lower() in ['m4a', 'wav', 'mp3']:
                wav_filepath = os.path.join(self.des_dir, f'{only_filename}.wav')
                convert_audio_to_wav(file_path, wav_filepath, minimize=self.minimize)
                ret.append(wav_filepath)
            else:
                print(f'failed to transform - {file_path}, not supported extension')
                raise NotImplemented

        return ret


class AddIntroEnding(AudioPipe):
    """
    오디오에 intro와 ending 음성을 추가하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: intro
        remove_input(bool): 입력물 삭제 여부, default: False
    """

    def __init__(self, des_dir: str = 'intro', remove_input: bool = False):
        super().__init__(des_dir, remove_input)

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            filename = get_filename(file_path)
            intro_filepath = os.path.join(self.des_dir, filename)
            add_intro_ending(file_path, intro_filepath)
            ret.append(intro_filepath)

        return ret


class SilentRemoverVAD(AudioPipe):
    """
    Vad 알고리즘을 활용하여 침묵을 제거하는 Pipe
    참고 : https://github.com/snakers4/silero-vad

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: silent
        remove_input(bool): 입력물 삭제 여부, default: False
        sampling_rate(int): vad 알고리즘에서 사용할 sampling rate, default: 16000
        threshold(float): 침묵 여부를 판가름 하는 threshold, default: 0.5
    """

    def __init__(self, des_dir: str = 'silent', remove_input: bool = False, sampling_rate: int = 16000,
                 threshold: float = 0.5):
        super().__init__(des_dir, remove_input)
        self.sampling_rate = sampling_rate
        self.threshold = threshold

    def next(self, filepath_list: List[str]) -> List[str]:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        ret = []
        for audio_file in filepath_list:
            filename = get_filename(audio_file)
            # Load your audio.
            audio = read_audio(os.path.join(data_path, audio_file), sampling_rate=self.sampling_rate)
            speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=self.sampling_rate,
                                                      threshold=self.threshold)
            # merge all speech chunks to one audio
            silent_filepath = os.path.join(self.des_dir, filename)
            save_audio(
                os.path.join(data_path, silent_filepath),
                collect_chunks(speech_timestamps, audio),
                sampling_rate=self.sampling_rate
            )
            del audio
            del speech_timestamps
            ret.append(silent_filepath)
        return ret


class SilentRemoverSimple(AudioPipe):
    """
    dBFS를 기준으로 침묵을 제거하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: silent_simple
        remove_input(bool): 입력물 삭제 여부, default: False
        silence_thresh_basis(int): 침묵 여부를 판가름하는 dBFS값
    """
    def __init__(self, des_dir: str = 'silent_simple', remove_input: bool = False, silence_thresh_basis: int = -20):
        super().__init__(des_dir, remove_input)
        self.silence_thresh_basis = silence_thresh_basis

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            filename = get_filename(file_path)
            silent_remover_filepath = os.path.join(self.des_dir, filename)

            # Load your audio.
            sound = AudioSegment.from_file(os.path.join(data_path, file_path))
            dbfs = sound.dBFS
            audio_chunks = split_on_silence(sound, min_silence_len=1000,
                                            silence_thresh=dbfs + self.silence_thresh_basis, keep_silence=100)
            print(f"{filename}'s dBFS : {dbfs}, chunk num : {len(audio_chunks)}")
            # Putting the file back together
            combined = AudioSegment.empty()
            for chunk in audio_chunks:
                combined += chunk

            combined.export(os.path.join(data_path, silent_remover_filepath), format=filename.split('.')[-1])

            ret.append(silent_remover_filepath)
        return ret


class AudioNormalizer(AudioPipe):  # dBFS
    """
    오디를 normalize하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: normal
        remove_input(bool): 입력물 삭제 여부, default: False
        method(str):
            normalize 하는 방법(basic/dBFS)
                basic: 전체 음성의 최대치와 최소치를 고려하여 normalize
                dBFS: {basis_dbfs} 값을 기준으로 normalize
            default: basic
        basis_dbfs(float): dBFS 방식으로 normailize 시 기준이 되는 값, default: -20.0
    """
    def __init__(self, des_dir: str = 'normal', remove_input: bool = False,
                 method: str = 'basic',
                 basis_dbfs: float = -20.0):
        super().__init__(des_dir, remove_input)
        self.method = method
        self.basis_dbfs = basis_dbfs

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            filename = get_filename(file_path)
            normal_filepath = os.path.join(self.des_dir, filename)
            if self.method == 'basic':
                convert_audio_to_(file_path, normal_filepath, format=filename.split('.')[-1], normalize=True,
                                  minimize=False, normal_method=self.method)
            elif self.method == 'dBFS':
                convert_audio_to_(file_path, normal_filepath, format=filename.split('.')[-1], normalize=True,
                                  minimize=False, basis_dbfs=self.basis_dbfs, normal_method=self.method)
            else:
                raise NotImplemented
            ret.append(normal_filepath)

        return ret

    def __repr__(self):
        return f'pipe - {type(self).__name__}({self.method})'


def split_audio(file_path, des_dir, min_length, max_length, min_silence, silence_thresh):
    filename = get_filename(file_path)
    sound = AudioSegment.from_file(os.path.join(data_path, file_path))
    total_length = len(sound)
    last_silence, chunk, process_length = 0, 0, 0
    only_filename = ''.join(filename.split('.')[:-1])
    file_extension = filename.split('.')[-1]
    ret = []

    silent_ranges = detect_silence(
        sound,
        # split on silences longer than 1000ms (1 sec)
        min_silence_len=min_silence,
        # anything under -16 dBFS is considered silence
        silence_thresh=silence_thresh
    )
    for range in silent_ranges:
        if total_length - process_length < min_length:
            filepath = os.path.join(data_path, des_dir, f'{only_filename}_{chunk}.{file_extension}')
            sound[last_silence:].export(filepath, format=file_extension)
            ret.append(os.path.join(des_dir, f'{only_filename}_{chunk}.{file_extension}'))
            break
        if range[1] - last_silence > max_length or (min_length < range[1] - last_silence < max_length):
            filepath = os.path.join(data_path, des_dir, f'{only_filename}_{chunk}.{file_extension}')
            sound[last_silence:range[1]].export(filepath, format=file_extension)
            ret.append(os.path.join(des_dir, f'{only_filename}_{chunk}.{file_extension}'))
            chunk = chunk + 1
            last_silence = range[1]
        process_length = range[1]
    if len(ret) > 0:
        return ret
    else:
        print(f'failed to split a file({file_path}), pass throw native file.')
        return [file_path]


class ToSplit(AudioPipe):
    """
    오디오를 침묵기준으로 분할하는 Pipe

    Args:
        des_dir(str): 결과를 저장할 디렉토리, default: split
        remove_input(bool): 입력물 삭제 여부, default: False
        min_length(int): 최소 시간(ms), default: 90s
        max_length(int): 최대 시간(ms), default: 120s
        min_silence(int): 음성을 자른 부분에 추가되는 묵음, default: 500ms
        silence_thresh(int): 침묵 여부를 판가름 하는 dBFS, default: -32
    """
    def __init__(self, des_dir: str = 'split', remove_input: bool = False,
                 min_length: int = 90 * 1000,
                 max_length: int = 120 * 1000,
                 min_silence: int = 500,
                 silence_thresh: int = -32):
        super().__init__(des_dir, remove_input)
        self.min_length = min_length
        self.max_length = max_length
        self.min_silence = min_silence
        self.silence_thresh = silence_thresh

    def next(self, filepath_list: List[str]) -> List[str]:
        ret = []
        for file_path in filepath_list:
            splitted_audio_path = split_audio(file_path, self.des_dir, self.min_length, self.max_length,
                                              self.min_silence, self.silence_thresh)
            ret.extend(splitted_audio_path)

        return ret


vad = SilentRemoverVAD(remove_input=True, threshold=0.3)


def preprocessing_simple() -> Pipechain:
    """
    wav(고용량)
    -> 노말라이즈(-20dBFS)
    -> 침묵 - simple(-20)
    -> 노말라이즈

    Returns:
         pipe(Pipechain): pipechain의 묶음
    """

    pipe_chain = Pipechain()
    pipe_chain.add(ToWav(minimize=False))
    pipe_chain.add(AudioNormalizer(des_dir='normal_dbfs', remove_input=True, method='dBFS'))
    pipe_chain.add(SilentRemoverSimple(des_dir='silent_simple', remove_input=True, silence_thresh_basis=-20))
    pipe_chain.add(AudioNormalizer(des_dir='normal_basic', remove_input=True, method='basic'))

    return pipe_chain


def preprocessing_vad() -> Pipechain:
    """
    wav(고용량)
    -> 노말라이즈(-20dBFS)
    -> 침묵(0.3)
    -> 노말라이즈

    Returns:
         pipe(Pipechain): pipechain의 묶음
    """

    pipe_chain = Pipechain()
    pipe_chain.add(ToWav(minimize=False))
    pipe_chain.add(AudioNormalizer(des_dir='normal_dbfs', remove_input=True, method='dBFS'))
    pipe_chain.add(vad)
    pipe_chain.add(AudioNormalizer(des_dir='normal_basic', remove_input=True, method='basic'))

    return pipe_chain


def preprocessing_vad_split() -> Pipechain:
    """
    wav(고용량)
    -> 노말라이즈(-20dBFS)
    -> 침묵(0.3)
    -> 노말라이즈
    -> split(4~5분)

    Returns:
         pipe(Pipechain): pipechain의 묶음
    """

    pipe_chain = Pipechain()
    pipe_chain.add(ToWav(minimize=False))
    pipe_chain.add(AudioNormalizer(des_dir='normal_dbfs', remove_input=True, method='dBFS'))
    pipe_chain.add(vad)
    pipe_chain.add(AudioNormalizer(des_dir='normal_basic', remove_input=True, method='basic'))
    pipe_chain.add(ToSplit(remove_input=True, min_length=240_000, max_length=300_000))  # 4분~5분

    return pipe_chain


def preprocessing_split() -> Pipechain:
    """
    wav(저용량)
    -> split(4~5분)

    Returns:
         pipe(Pipechain): pipechain의 묶음
    """
    pipe_chain = Pipechain()
    pipe_chain.add(ToWav(minimize=True))
    pipe_chain.add(ToSplit(remove_input=True, min_length=240_000, max_length=300_000))  # 4분~5분

    return pipe_chain
