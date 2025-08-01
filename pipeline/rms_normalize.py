import torch
import numpy as np
from config import TARGET_RMS_DB, RMS_EPSILON, MAX_GAIN_DB, COMPRESSION_THRESHOLD

def calculate_rms(audio):
    """
    오디오의 RMS 값을 계산합니다.
    
    :param audio: 오디오 데이터 (torch.Tensor 또는 numpy.ndarray)
    :return: RMS 값
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    rms = np.sqrt(np.mean(audio ** 2))
    return rms

def rms_to_db(rms):
    """
    RMS 값을 dB로 변환합니다.
    
    :param rms: RMS 값
    :return: dB 값
    """
    return 20 * np.log10(rms + RMS_EPSILON)

def db_to_rms(db):
    """
    dB 값을 RMS로 변환합니다.
    
    :param db: dB 값
    :return: RMS 값
    """
    return 10 ** (db / 20)

def normalize_rms(audio, target_db=TARGET_RMS_DB):
    """
    오디오의 RMS를 목표 dB 레벨로 정규화합니다.
    
    :param audio: 입력 오디오 데이터 (torch.Tensor 또는 numpy.ndarray)
    :param target_db: 목표 RMS 레벨 (dB)
    :return: 정규화된 오디오 데이터 (입력과 같은 타입)
    """
    is_tensor = isinstance(audio, torch.Tensor)
    
    if is_tensor:
        audio_np = audio.numpy()
    else:
        audio_np = audio
    
    # 현재 RMS 계산
    current_rms = calculate_rms(audio_np)
    current_db = rms_to_db(current_rms)
    
    # 목표 RMS 계산
    target_rms = db_to_rms(target_db)
    
    # 정규화 팩터 계산
    if current_rms > RMS_EPSILON:
        scaling_factor = target_rms / current_rms
    else:
        scaling_factor = 1.0
    
    # 오디오 정규화
    normalized_audio = audio_np * scaling_factor
    
    print(f"🔊 RMS 정규화: {current_db:.2f}dB -> {target_db:.2f}dB (스케일링: {scaling_factor:.4f})")
    
def adaptive_level_adjust(audio, target_rms_db=TARGET_RMS_DB, max_gain_db=MAX_GAIN_DB, compression_threshold=COMPRESSION_THRESHOLD):
    """
    적응적 레벨 조정: 작은 소리는 증폭, 큰 소리는 압축
    
    :param audio: 입력 오디오 (torch.Tensor 또는 numpy.ndarray)
    :param target_rms_db: 목표 RMS 레벨 (dB)
    :param max_gain_db: 최대 증폭 게인 (dB)
    :param compression_threshold: 압축 시작 임계값 (0~1)
    :return: 조정된 오디오 (입력과 같은 타입)
    """
    is_tensor = isinstance(audio, torch.Tensor)
    
    if is_tensor:
        audio_np = audio.numpy()
    else:
        audio_np = audio.copy()
    
    if len(audio_np) == 0:
        return audio
    
    # 현재 RMS 계산
    current_rms = np.sqrt(np.mean(audio_np ** 2))
    
    if current_rms < 1e-8:  # 무음에 가까운 경우
        print(f"⚠️ Nearly silent audio (RMS: {current_rms:.8f}), skipping adjustment")
        return audio
    
    # 목표 RMS 레벨
    target_rms = 10 ** (target_rms_db / 20.0)
    current_rms_db = 20 * np.log10(current_rms + 1e-8)
    
    # 최대값 확인 (클리핑 방지용)
    max_val = np.max(np.abs(audio_np))
    
    if current_rms < target_rms:
        # 🔊 작은 소리: 증폭
        gain_factor = target_rms / current_rms
        
        # 최대 증폭 제한
        max_gain_factor = 10 ** (max_gain_db / 20.0)
        gain_factor = min(gain_factor, max_gain_factor)
        
        adjusted_audio = audio_np * gain_factor
        
        # 클리핑 방지
        new_max = np.max(np.abs(adjusted_audio))
        if new_max > 0.95:
            adjusted_audio = adjusted_audio * (0.95 / new_max)
            actual_gain = 20 * np.log10((0.95 / new_max) * gain_factor)
        else:
            actual_gain = 20 * np.log10(gain_factor)
            
        print(f"🔊 Amplified: {current_rms_db:.1f}dB → {target_rms_db:.1f}dB (+{actual_gain:.1f}dB)")
        
    elif max_val > compression_threshold:
        # 🔇 큰 소리: 소프트 압축 (리미팅)
        # Soft knee compression
        ratio = 3.0  # 압축 비율
        threshold = compression_threshold
        
        # 압축 적용
        adjusted_audio = np.copy(audio_np)
        over_threshold = np.abs(adjusted_audio) > threshold
        
        # 압축 함수: 임계값을 넘는 부분을 소프트하게 압축
        over_amount = np.abs(adjusted_audio[over_threshold]) - threshold
        compressed_over = threshold + over_amount / ratio
        
        # 원래 부호 유지
        adjusted_audio[over_threshold] = np.sign(adjusted_audio[over_threshold]) * compressed_over
        
        # RMS를 목표 레벨로 조정
        new_rms = np.sqrt(np.mean(adjusted_audio ** 2))
        if new_rms > 1e-8:
            rms_adjust = target_rms / new_rms
            adjusted_audio = adjusted_audio * rms_adjust
        
        final_rms_db = 20 * np.log10(np.sqrt(np.mean(adjusted_audio ** 2)) + 1e-8)
        print(f"🔇 Compressed & normalized: {current_rms_db:.1f}dB → {final_rms_db:.1f}dB")
        
    else:
        # 📊 적절한 범위: 약간의 조정만
        gain_factor = target_rms / current_rms
        # 제한된 조정 (±3dB)
        gain_factor = np.clip(gain_factor, 10**(-3/20), 10**(3/20))
        
        adjusted_audio = audio_np * gain_factor
        final_rms_db = 20 * np.log10(np.sqrt(np.mean(adjusted_audio ** 2)) + 1e-8)
        print(f"📊 Minor adjustment: {current_rms_db:.1f}dB → {final_rms_db:.1f}dB")
    
    # 원래 타입으로 반환
    if is_tensor:
        return torch.from_numpy(adjusted_audio).float()
    else:
        return adjusted_audio
