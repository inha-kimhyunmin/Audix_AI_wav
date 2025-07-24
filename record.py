import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, SEGMENT_DURATION

def record_segment():
    """
    오디오 세그먼트를 녹음합니다.
    :return: 녹음된 오디오 데이터 (numpy 배열)
    """
    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']  # type: ignore
    print(f"🎙 녹음 시작 (채널: {channels})")
    audio = sd.rec(int(SAMPLE_RATE * SEGMENT_DURATION), samplerate=SAMPLE_RATE,
                   channels=channels, dtype='float32')
    sd.wait()
    return audio  # [samples, channels]
