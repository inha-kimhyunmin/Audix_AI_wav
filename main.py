"""
메인 실행 파일: WAV 파일 전처리 + ONNX 분류 분석을 순차적으로 실행
"""

import os
import sys
from datetime import datetime

# 1단계: .pt 파일 생성 (audio_preprocessing.py)
from audio_preprocessing import process_wav_file, load_model
from resample import init_resampler

# 2단계: .pt 파일 분석 (integrated_analysis.py)  
from integrated_analysis import process_pt_files_with_classification

def main_pipeline(wav_file_path, target_parts, onnx_model_base_path="ResNet18_onnx", device_name="machine_001"):
    """
    완전한 파이프라인: WAV 파일 → .pt 파일 생성 → ONNX 분류 분석
    
    Args:
        wav_file_path: 입력 WAV 파일 경로
        target_parts: 분석할 부품 리스트 (예: ['fan', 'pump'])
        onnx_model_base_path: ONNX 모델들이 저장된 폴더 경로
        device_name: 장치명
    
    Returns:
        dict: 최종 분석 결과
    """
    
    print("🚀 통합 분석 파이프라인 시작")
    print("="*60)
    
    try:
        # === 1단계: .pt 파일 생성 ===
        print("📋 1단계: WAV 파일에서 .pt 파일 생성")
        print(f"📁 입력 파일: {wav_file_path}")
        print(f"🎯 대상 부품: {target_parts}")
        
        # Demucs 모델 로드
        print("🔧 Demucs 모델 로딩 중...")
        model, source_names = load_model()
        init_resampler(model.samplerate)
        
        # .pt 파일 생성
        generated_files = process_wav_file(model, source_names, wav_file_path, target_parts=target_parts)
        
        if not generated_files:
            raise ValueError("❌ .pt 파일이 생성되지 않았습니다.")
        
        print(f"✅ 1단계 완료: {len(generated_files)}개 .pt 파일 생성")
        for file_path in generated_files:
            print(f"  📄 {file_path}")
        
        # === 2단계: .pt 파일 분석 ===
        print(f"\n📋 2단계: 각 부품별 전용 ONNX 모델로 분류 분석")
        print(f"🤖 ONNX 모델 폴더: {onnx_model_base_path}")
        
        analysis_results = process_pt_files_with_classification(
            pt_files=generated_files,
            onnx_model_base_path=onnx_model_base_path,
            device_name=device_name
        )
        
        print(f"✅ 2단계 완료: {analysis_results['total_parts']}개 부품 분석")
        
        # === 최종 결과 통합 ===
        final_result = {
            "pipeline_info": {
                "input_wav_file": wav_file_path,
                "target_parts": target_parts,
                "generated_pt_files": generated_files,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "analysis_results": analysis_results
        }
        
        return final_result
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        raise

def print_final_results(results):
    """결과를 보기 좋게 출력합니다."""
    
    print("\n" + "="*60)
    print("🎯 최종 통합 분석 결과")
    print("="*60)
    
    # 파이프라인 정보
    pipeline_info = results["pipeline_info"]
    print(f"📁 입력 파일: {pipeline_info['input_wav_file']}")
    print(f"🎯 분석 대상: {pipeline_info['target_parts']}")
    print(f"📄 생성된 .pt 파일: {len(pipeline_info['generated_pt_files'])}개")
    print(f"⏰ 처리 시간: {pipeline_info['timestamp']}")
    
    # 분석 결과
    analysis = results["analysis_results"]
    print(f"\n🏭 장치: {analysis['device_name']}")
    print(f"📊 분석 부품 수: {analysis['total_parts']}")
    print(f"⚠️ 이상 감지 부품: {analysis['anomaly_count']}")
    
    print("\n📋 상세 결과:")
    for result in analysis['results']:
        status = "🚨 이상" if result['anomaly_detected'] else "✅ 정상"
        model_info = f", 모델: {result['model_used']}" if 'model_used' in result else ""
        print(f"  {result['part_name']}: {status} "
              f"(확률: {result['anomaly_probability']:.3f}{model_info})")
    
    # 생성된 파일들
    print(f"\n📄 생성된 .pt 파일들:")
    for file_path in pipeline_info['generated_pt_files']:
        print(f"  📄 {file_path}")

def save_results_to_json(results, output_filename=None):
    """결과를 JSON 파일로 저장합니다."""
    
    if output_filename is None:
        device_name = results["analysis_results"]["device_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pipeline_result_{device_name}_{timestamp}.json"
    
    import json
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 결과가 {output_filename}에 저장되었습니다.")
    return output_filename

# === 메인 실행 부분 ===
if __name__ == "__main__":
    # 설정값들
    WAV_FILE = "test01/mixture.wav"  # 입력 WAV 파일
    TARGET_PARTS = ["fan", "pump", "slider", "gearbox", "bearing"]   # 분석할 부품들
    ONNX_MODEL_BASE_PATH = "ResNet18_onnx"  # ONNX 모델들이 저장된 폴더
    DEVICE_NAME = "machine_001"      # 장치명
    
    try:
        # 통합 파이프라인 실행
        results = main_pipeline(
            wav_file_path=WAV_FILE,
            target_parts=TARGET_PARTS,
            onnx_model_base_path=ONNX_MODEL_BASE_PATH,
            device_name=DEVICE_NAME
        )
        
        # 결과 출력
        print_final_results(results)
        
        # JSON 파일로 저장
        save_results_to_json(results)
        
        print("\n🎉 모든 과정이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 메인 파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
