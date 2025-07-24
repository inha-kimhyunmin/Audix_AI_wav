import sounddevice as sd
import torch
from config import SAMPLE_RATE, NOISE_SAMPLE_PATH

def record_noise_sample(duration=10):
    """
    배경 잡음 샘플을 녹음하고 저장합니다.
    :param duration: 녹음 시간 (초)
    :return: None
    """
    print(f"🔇 {duration}초간 배경 잡음 녹음...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()
    torch.save(torch.tensor(audio), NOISE_SAMPLE_PATH)
    print(f"✅ 저장 완료: {NOISE_SAMPLE_PATH}")

if __name__ == "__main__":
    record_noise_sample()
