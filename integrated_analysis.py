import os
import json
import glob
from onnx import predict_single_file_onnx_json

def process_pt_files_with_classification(pt_files, onnx_model_path, device_name="unknown_device"):
    """
    기존 .pt 파일들을 ONNX 모델로 분류하는 함수
    
    Args:
        pt_files: 분석할 .pt 파일 경로 리스트
        onnx_model_path: ONNX 분류 모델 경로  
        device_name: 장치명
    
    Returns:
        dict: 분석 결과를 담은 딕셔너리
    """
    
    # 각 .pt 파일에 대해 ONNX 분류 수행
    classification_results = []
    
    for pt_file_path in pt_files:
        # 파일명에서 부품명 추출 (예: output/2025-07-25_01-48-11_mic_1_fan.pt -> fan)
        filename = os.path.basename(pt_file_path)
        part_name = filename.split('_')[-1].replace('.pt', '')
        
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
            "part_name": part_name,
            "pt_file_path": pt_file_path,
            "device_name": device_name,
            "anomaly_detected": classification_result["result"],
            "anomaly_probability": classification_result["probability"]
        }
        
        classification_results.append(integrated_result)
        
        print(f"✅ {part_name}: {'이상 감지' if classification_result['result'] else '정상'} "
              f"(확률: {classification_result['probability']:.3f})")
    
    # 분석할 부품명들 추출
    analyzed_parts = [result["part_name"] for result in classification_results]
    
    # 최종 결과 정리
    final_result = {
        "device_name": device_name,
        "analyzed_parts": analyzed_parts,
        "total_parts": len(classification_results),
        "anomaly_count": sum(1 for r in classification_results if r["anomaly_detected"]),
        "results": classification_results
    }
    
    return final_result

def analyze_pt_files_by_pattern(output_dir="output", onnx_model_path="ResNet18_onnx/fold0_best_model.onnx", device_name="machine_001"):
    """
    output 폴더에서 .pt 파일들을 찾아서 분석합니다.
    
    Args:
        output_dir: .pt 파일들이 저장된 디렉토리
        onnx_model_path: ONNX 모델 경로
        device_name: 장치명
    
    Returns:
        dict: 분석 결과
    """
    # output 폴더에서 .pt 파일들 찾기
    pt_pattern = os.path.join(output_dir, "*.pt")
    pt_files = glob.glob(pt_pattern)
    
    if not pt_files:
        raise FileNotFoundError(f"❌ {output_dir} 폴더에 .pt 파일이 없습니다.")
    
    print(f"📁 발견된 .pt 파일들: {len(pt_files)}개")
    for pt_file in pt_files:
        print(f"  📄 {pt_file}")
    
    return process_pt_files_with_classification(pt_files, onnx_model_path, device_name)

# === 사용 예시 ===
if __name__ == "__main__":
    # 설정
    ONNX_MODEL = "ResNet18_onnx/fold0_best_model.onnx"  # 분류용 ONNX 모델
    DEVICE_NAME = "machine_001"
    
    try:
        # .pt 파일들 분석 실행
        results = analyze_pt_files_by_pattern(
            output_dir="output",
            onnx_model_path=ONNX_MODEL,
            device_name=DEVICE_NAME
        )
        
        # 결과 출력
        print("\n" + "="*60)
        print("🎯 최종 분석 결과")
        print("="*60)
        print(f"🏭 장치: {results['device_name']}")
        print(f"🎯 분석 대상: {results['analyzed_parts']}")
        print(f"📊 분석 부품 수: {results['total_parts']}")
        print(f"⚠️ 이상 감지 부품: {results['anomaly_count']}")
        
        print("\n📋 상세 결과:")
        for result in results['results']:
            status = "🚨 이상" if result['anomaly_detected'] else "✅ 정상"
            print(f"  {result['part_name']}: {status} "
                  f"(확률: {result['anomaly_probability']:.3f})")
        
        # JSON 파일로 저장
        output_file = f"analysis_result_{DEVICE_NAME}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
