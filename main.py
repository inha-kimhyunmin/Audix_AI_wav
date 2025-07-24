from config import CHANNEL_PARTS, SAMPLE_RATE
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

def process_wav_file(model, source_names, wav_path):
    """
    WAV 파일을 처리하고 결과를 JSON으로 반환합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트 (예: ['fan', 'pump', ...])
    :param wav_path: 입력 WAV 파일 경로
    :return: 처리 결과 JSON
    """
    print(f"\n🎵 WAV 파일 처리 시작: {wav_path}")
    
    # WAV 파일 로드
    start_load = time.time()
    audio = load_wav_file(wav_path)
    end_load = time.time()
    print(f"📂 파일 로드 시간: {(end_load - start_load):.2f}초")
    
    # 타임스탬프 생성
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 결과 저장용 리스트
    results = []
    
    # 스테레오로 변환 (2채널로 복사)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)  # [2, samples]
    
    # 각 마이크(채널)별 처리
    for mic_idx in range(audio.shape[0]):
        print(f"\n🎧 마이크 {mic_idx + 1} 처리 중")
        start_total = time.time()
        
        # 1. 적응적 레벨 조정 (작은 소리는 증폭, 큰 소리는 압축)
        start_normalize = time.time()
        current_rms = calculate_rms(audio[mic_idx])
        current_db = rms_to_db(current_rms)
        print(f"📊 원본 오디오 RMS: {current_db:.2f}dB")
        
        normalized_audio = adaptive_level_adjust(audio[mic_idx])
        
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
        
        # 3. 저장 (noise는 제외)
        start_save = time.time()
        parts_for_mic = CHANNEL_PARTS[mic_idx]
        for src_idx, src in enumerate(sources):
            if src_idx < len(source_names):
                src_name = source_names[src_idx]
                
                # noise는 저장하지 않음
                if src_name.lower() == "noise":
                    print(f"🗑️ Noise 소스 무시: {src_name}")
                    continue
                
                file_path = save_mel_tensor(src, mic_idx + 1, src_name, timestamp_str, parts_for_mic)
                
                if file_path:  # 저장이 성공한 경우
                    result_entry = {
                        "mic_number": mic_idx + 1,
                        "part_name": src_name,
                        "file_path": file_path
                    }
                    results.append(result_entry)
                    print(f"📁 저장 완료: {src_name}")
        
        # 분리된 소스 수 정보 출력
        total_sources = len(sources)
        saved_sources = len([s for s in source_names[:total_sources] if s.lower() != "noise"])
        print(f"🎛️ 총 {total_sources}개 소스 분리 완료 (저장: {saved_sources}개, 무시: {total_sources - saved_sources}개)")
        
        end_save = time.time()
        print(f"💾 저장 시간: {(end_save - start_save):.2f}초")
        
        # 총 시간
        end_total = time.time()
        print(f"⏱️ 마이크 {mic_idx + 1} 전체 처리 시간: {(end_total - start_total):.2f}초")
    
    # JSON 결과 반환
    result_json = {
        "input_file": wav_path,
        "timestamp": timestamp_str,
        "processed_files": results
    }
    
    return result_json

def process_multiple_wav_files(model, source_names, wav_paths):
    """
    여러 WAV 파일을 배치로 처리합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트
    :param wav_paths: WAV 파일 경로 리스트
    :return: 모든 처리 결과를 포함한 JSON
    """
    all_results = []
    
    for i, wav_path in enumerate(wav_paths):
        print(f"\n🔄 파일 {i+1}/{len(wav_paths)} 처리 중: {wav_path}")
        try:
            result = process_wav_file(model, source_names, wav_path)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 파일 {wav_path} 처리 중 오류 발생: {e}")
            error_result = {
                "input_file": wav_path,
                "error": str(e),
                "processed_files": []
            }
            all_results.append(error_result)
    
    final_result = {
        "batch_processing": True,
        "total_files": len(wav_paths),
        "successful_files": len([r for r in all_results if "error" not in r]),
        "results": all_results
    }
    
    return final_result
    """
    오디오 스트림을 처리하고 저장합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트 (예: ['fan', 'pump', ...])
    :param repeat: 반복 횟수
    """
    noise_clip = load_noise_clip()

    for i in range(repeat):
        print(f"\n📡 반복 {i+1}/{repeat}")

        start_record = time.time()
        audio = record_segment()
        end_record = time.time()
        print(f"🎙️ 녹음 완료: {(end_record - start_record):.2f}초")
        print("audio.shape = ", audio.shape)
        print(f"audio = {audio}, type : {type(audio)}")

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for mic_idx in range(audio.shape[1]):
            print(f"\n🎧 마이크 {mic_idx + 1} 처리 중")
            start_total = time.time()

            # 1. 잡음 제거
            start_denoise = time.time()
            clean = denoise(audio[:, mic_idx], noise_clip)
            end_denoise = time.time()
            print(f"clean : {clean}, type : {type(clean)}")
            print(f"🧹 잡음 제거 시간: {(end_denoise - start_denoise):.2f}초")

            # 2. 분리
            start_sep = time.time()
            sources = separate(model, clean)
            end_sep = time.time()
            print(f"sources = {sources}, type : {type(sources)}")
            print(f"🎛️ 소리 분리 시간: {(end_sep - start_sep):.2f}초")

            # 3. 저장
            start_save = time.time()
            parts_for_mic = CHANNEL_PARTS[mic_idx]
            for src_idx, src in enumerate(sources):
                src_name = source_names[src_idx]
                mel_result = save_mel_tensor(src, mic_idx + 1, src_name, timestamp_str, parts_for_mic)
                print(f"📁 저장 완료: {src_name}")
            end_save = time.time()
            print(f"💾 저장 시간: {(end_save - start_save):.2f}초")

            # 총 시간
            end_total = time.time()
            print(f"⏱️ 마이크 {mic_idx + 1} 전체 처리 시간: {(end_total - start_total):.2f}초")

if __name__ == "__main__":
    model, source_names = load_model()
    init_resampler(model.samplerate)
    
    # 예시: WAV 파일 처리
    wav_file_path = "test01/mix.wav"  # 처리할 WAV 파일 경로
    
    try:
        result = process_wav_file(model, source_names, wav_file_path)
        
        # 결과를 JSON으로 출력
        print("\n📊 처리 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 선택적으로 JSON 파일로 저장
        json_filename = f"result_{result['timestamp']}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 결과 JSON 파일 저장: {json_filename}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
