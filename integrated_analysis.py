import os
import json
from main import process_wav_file, load_model  # 기존 main.py 함수들
from onnx import predict_single_file_onnx_json

def process_audio_with_classification(wav_file_path, onnx_model_path, mic_number, device_name="unknown_device"):
    """
    WAV 파일을 Demucs로 분리하고 각 부품별로 ONNX 모델로 분류하는 통합 함수
    
    Args:
        wav_file_path: 입력 WAV 파일 경로
        onnx_model_path: ONNX 분류 모델 경로
        mic_number: 마이크 번호
        device_name: 장치명
    
    Returns:
        dict: 전체 결과를 담은 딕셔너리
    """
    
    # 1. Demucs 모델 로드
    print("🔧 Demucs 모델 로딩 중...")
    model, source_names = load_model()
    
    # 2. Demucs로 소스 분리 및 mel spectrogram 생성
    print("🎵 Demucs 소스 분리 및 mel spectrogram 생성 중...")
    separation_result = process_wav_file(model, source_names, wav_file_path)
    
    # 3. 각 분리된 소스에 대해 ONNX 분류 수행
    classification_results = []
    
    # separation_result에서 processed_files 리스트 가져오기
    processed_files = separation_result.get("processed_files", [])
    
    for result in processed_files:
        part_name = result["part_name"]
        pt_file_path = result["file_path"]
        
        print(f"🤖 {part_name} 분류 중...")
        
        # ONNX 모델로 분류
        classification_result = predict_single_file_onnx_json(
            onnx_model_path=onnx_model_path,
            pt_file_path=pt_file_path,
            device_name=device_name,
            in_ch=1,  # mel spectrogram 채널 수
            threshold=0.5
        )
        
        # 결과 통합
        integrated_result = {
            "mic_number": mic_number,
            "part_name": part_name,
            "pt_file_path": pt_file_path,
            "device_name": classification_result["device_name"],
            "anomaly_detected": classification_result["result"],
            "anomaly_probability": classification_result["probability"]
        }
        
        classification_results.append(integrated_result)
        
        print(f"✅ {part_name}: {'이상 감지' if classification_result['result'] else '정상'} "
              f"(확률: {classification_result['probability']:.3f})")
    
    # 3. 전체 결과 정리
    final_result = {
        "input_wav_file": wav_file_path,
        "mic_number": mic_number,
        "device_name": device_name,
        "total_parts": len(classification_results),
        "anomaly_count": sum(1 for r in classification_results if r["anomaly_detected"]),
        "results": classification_results
    }
    
    return final_result

# === 사용 예시 ===
if __name__ == "__main__":
    # 설정
    WAV_FILE = "test01/mixture.wav"  # 입력 WAV 파일
    ONNX_MODEL = "ResNet18_onnx/fold0_best_model.onnx"  # 분류용 ONNX 모델
    MIC_NUMBER = 1
    DEVICE_NAME = "machine_001"
    
    try:
        # 통합 처리 실행
        results = process_audio_with_classification(
            wav_file_path=WAV_FILE,
            onnx_model_path=ONNX_MODEL,
            mic_number=MIC_NUMBER,
            device_name=DEVICE_NAME
        )
        
        # 결과 출력
        print("\n" + "="*60)
        print("🎯 최종 분석 결과")
        print("="*60)
        print(f"📁 입력 파일: {results['input_wav_file']}")
        print(f"🎤 마이크: {results['mic_number']}")
        print(f"🏭 장치: {results['device_name']}")
        print(f"📊 분석 부품 수: {results['total_parts']}")
        print(f"⚠️ 이상 감지 부품: {results['anomaly_count']}")
        
        print("\n📋 상세 결과:")
        for result in results['results']:
            status = "🚨 이상" if result['anomaly_detected'] else "✅ 정상"
            print(f"  {result['part_name']}: {status} "
                  f"(확률: {result['anomaly_probability']:.3f})")
        
        # JSON 파일로 저장
        output_file = f"analysis_result_{MIC_NUMBER}_{DEVICE_NAME}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
