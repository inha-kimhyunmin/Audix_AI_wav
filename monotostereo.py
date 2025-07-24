# audio_utils.py
import torch
import numpy as np

def mono_to_stereo(audio):
    """
    1채널 오디오를 2채널로 복제하고 (1, 2, samples) 형태로 반환합니다.
    
    :param audio: (samples,) 또는 (samples, 1) 모노 오디오 numpy 배열 또는 torch 텐서
    :return: (1, 2, samples) 형태의 torch.Tensor
    """
    if isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            audio = audio.unsqueeze(1)  # (samples,) → (samples, 1)
        stereo = audio.repeat(1, 2)     # (samples, 2)
        stereo = stereo.transpose(0, 1).unsqueeze(0)  # (samples, 2) → (2, samples) → (1, 2, samples)
        return stereo

    elif isinstance(audio, np.ndarray):
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)  # (samples,) → (samples, 1)
        stereo = np.repeat(audio, 2, axis=1)       # (samples, 2)
        stereo = np.transpose(stereo, (1, 0))      # (2, samples)
        stereo = np.expand_dims(stereo, axis=0)    # (1, 2, samples)
        return torch.tensor(stereo, dtype=torch.float32)

    else:
        raise TypeError("audio must be either a torch.Tensor or a numpy.ndarray")
