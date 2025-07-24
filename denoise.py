import torch
import noisereduce as nr
from config import SAMPLE_RATE, NOISE_SAMPLE_PATH

def load_noise_clip():
    """
    배경 잡음 샘플을 로드합니다.
    :return: 배경 잡음 샘플 (numpy 배열)
    """
    return torch.load(NOISE_SAMPLE_PATH, weights_only=True).numpy()

def denoise(audio_np, noise_clip=None):
    """
    오디오 데이터를 잡음 제거합니다.
    :param audio_np: 입력 오디오 데이터 (numpy 배열)
    :param noise_clip: 배경 잡음 샘플 (numpy 배열), None이면 기본 잡음 제거
    :return: 잡음 제거된 오디오 데이터 (numpy 배열)
    """
    if noise_clip is None:
        return nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
    return nr.reduce_noise(y=audio_np, y_noise=noise_clip, sr=SAMPLE_RATE)
