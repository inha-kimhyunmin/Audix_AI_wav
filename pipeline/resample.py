# === resample.py ===
import torch
from torchaudio.transforms import Resample
from config import SAMPLE_RATE, DEVICE

RESAMPLER = None  # 전역 변수로 사용

def init_resampler(model_samplerate):
    """
    모델의 샘플레이트와 입력 샘플레이트(SAMPLE_RATE)가 다를 경우 리샘플러 초기화

    :param model_samplerate: 모델이 훈련된 샘플레이트
    :return: torch.nn.Module 또는 None
    """
    global RESAMPLER

    if SAMPLE_RATE != model_samplerate:
        RESAMPLER = Resample(SAMPLE_RATE, model_samplerate).to(DEVICE)
        print(f"🎚️ 리샘플러 생성됨: {SAMPLE_RATE} → {model_samplerate}")
    else:
        RESAMPLER = None
        print("✅ 리샘플러 불필요 (샘플레이트 일치)")

    return RESAMPLER


def maybe_resample(audio):
    """
    전역 RESAMPLER가 정의되어 있다면 오디오를 리샘플링함

    :param audio: (B, C, T) 형태의 torch.Tensor
    :return: 리샘플링된 또는 원본 오디오
    """
    global RESAMPLER

    if RESAMPLER is not None:
        return RESAMPLER(audio)
    return audio
