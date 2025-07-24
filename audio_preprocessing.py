from config import SAMPLE_RATE
from model import load_model, separate
from mel import save_mel_tensor
from rms_normalize import adaptive_level_adjust, calculate_rms, rms_to_db
from datetime import datetime
from resample import init_resampler
import time
import json
import torchaudio
import torch

def load_wav_file(wav_path):
    """
    WAV 파일을 로드합니다.
    
    :param wav_path: WAV 파일 경로
    :return: 오디오 텐서 (shape: [channels, samples])
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # 샘플링 레이트 확인
    if sample_rate != SAMPLE_RATE:
        print(f"⚠️ 샘플링 레이트 불일치: {sample_rate}Hz -> {SAMPLE_RATE}Hz로 리샘플링")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # 모노 채널로 변환 (요청사항에 따라)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 10초 길이로 맞추기 (패딩 또는 자르기)
    target_length = SAMPLE_RATE * 10  # 10초
    current_length = waveform.shape[1]
    
    if current_length < target_length:
        # 패딩
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_length > target_length:
        # 자르기
        waveform = waveform[:, :target_length]
    
    print(f"📁 WAV 파일 로드 완료: {wav_path}")
    print(f"🔊 오디오 형태: {waveform.shape} (샘플링 레이트: {SAMPLE_RATE}Hz)")
    
    return waveform

def process_wav_file(model, source_names, wav_path, target_parts=None):
    """
    WAV 파일을 처리하고 .pt 파일들을 생성합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트 (예: ['fan', 'pump', ...])
    :param wav_path: 입력 WAV 파일 경로
    :param target_parts: 분석할 부품 리스트 (예: ['fan', 'pump']) - None이면 모든 부품 처리
    :return: 생성된 .pt 파일 경로들
    """
    print(f"\n🎵 WAV 파일 처리 시작: {wav_path}")
    
    # 타겟 부품이 지정되지 않으면 noise를 제외한 모든 부품 처리
    if target_parts is None:
        target_parts = [src for src in source_names if src.lower() != "noise"]
    
    print(f"🎯 분석 대상 부품: {target_parts}")
    
    # 유효한 부품인지 확인
    available_parts = [src for src in source_names if src.lower() != "noise"]
    invalid_parts = [part for part in target_parts if part not in available_parts]
    if invalid_parts:
        raise ValueError(f"❌ 지원하지 않는 부품: {invalid_parts}. 사용 가능한 부품: {available_parts}")
    
    # WAV 파일 로드
    start_load = time.time()
    audio = load_wav_file(wav_path)
    end_load = time.time()
    print(f"📂 파일 로드 시간: {(end_load - start_load):.2f}초")
    
    # 타임스탬프 생성
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 결과 저장용 리스트
    generated_files = []
    
    # 1. 적응적 레벨 조정 (작은 소리는 증폭, 큰 소리는 압축)
    start_normalize = time.time()
    current_rms = calculate_rms(audio.squeeze())  # 모노 채널이므로 squeeze 사용
    current_db = rms_to_db(current_rms)
    print(f"📊 원본 오디오 RMS: {current_db:.2f}dB")
    
    normalized_audio = adaptive_level_adjust(audio.squeeze())
    
    # adaptive_level_adjust가 tensor를 반환할 수 있으므로 numpy로 변환
    if isinstance(normalized_audio, torch.Tensor):
        normalized_audio = normalized_audio.numpy()
    
    end_normalize = time.time()
    print(f"🔧 적응적 레벨 조정 시간: {(end_normalize - start_normalize):.2f}초")
    
    # 2. 분리
    start_sep = time.time()
    sources = separate(model, normalized_audio)
    end_sep = time.time()
    print(f"🎛️ 소리 분리 시간: {(end_sep - start_sep):.2f}초")
    
    # 3. 저장 (target_parts에 있는 부품만 저장)
    start_save = time.time()
    saved_count = 0
    
    for src_idx, src in enumerate(sources):
        if src_idx < len(source_names):
            src_name = source_names[src_idx]
            
            # target_parts에 없는 부품은 건너뛰기
            if src_name not in target_parts:
                print(f"⏭️ {src_name} 건너뛰기 (분석 대상 아님)")
                continue
            
            # noise는 이미 target_parts에서 제외됨
            file_path = save_mel_tensor(src, 1, src_name, timestamp_str, target_parts)  # 마이크 번호를 1로 고정
            
            if file_path:  # 저장이 성공한 경우
                generated_files.append(file_path)
                saved_count += 1
                print(f"📁 저장 완료: {src_name}")
    
    # 분리된 소스 수 정보 출력
    total_sources = len(sources)
    print(f"🎛️ 총 {total_sources}개 소스 분리 완료 (저장: {saved_count}개, 건너뛰기: {total_sources - saved_count}개)")
    
    end_save = time.time()
    print(f"💾 저장 시간: {(end_save - start_save):.2f}초")
    
    # 생성된 .pt 파일 경로들 반환
    return generated_files

def process_multiple_wav_files(model, source_names, wav_paths, target_parts=None):
    """
    여러 WAV 파일을 배치로 처리해서 .pt 파일들을 생성합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트
    :param wav_paths: WAV 파일 경로 리스트
    :param target_parts: 분석할 부품 리스트
    :return: 모든 생성된 .pt 파일 경로들
    """
    all_generated_files = []
    
    for i, wav_path in enumerate(wav_paths):
        print(f"\n🔄 파일 {i+1}/{len(wav_paths)} 처리 중: {wav_path}")
        try:
            generated_files = process_wav_file(model, source_names, wav_path, target_parts=target_parts)
            all_generated_files.extend(generated_files)
        except Exception as e:
            print(f"❌ 파일 {wav_path} 처리 중 오류 발생: {e}")
    
    return all_generated_files


if __name__ == "__main__":
    model, source_names = load_model()
    init_resampler(model.samplerate)
    
    # 예시: WAV 파일 처리
    wav_file_path = "test01/mixture.wav"  # 처리할 WAV 파일 경로
    target_parts = ["fan", "pump"]  # 이 WAV 파일에 포함된 부품들 (실제 상황에 맞게 수정)
    
    try:
        generated_files = process_wav_file(model, source_names, wav_file_path, target_parts=target_parts)
        
        # 결과 출력
        print("\n📊 생성된 .pt 파일들:")
        for file_path in generated_files:
            print(f"  ✅ {file_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
