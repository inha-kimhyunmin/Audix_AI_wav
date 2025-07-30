"""
메인 실행 파일: WAV 파일 전처리 + ONNX 분류 분석을 순차적으로 실행
"""

import os
import sys
import json
import csv
import glob
from datetime import datetime
from pathlib import Path

# 1단계: .pt 파일 생성 (audio_preprocessing.py)
from audio_preprocessing import process_wav_file, load_model
from resample import init_resampler

# 2단계: .pt 파일 분석 (integrated_analysis.py)  
from integrated_analysis import process_pt_files_with_classification

def main_pipeline(wav_file_path, target_parts, onnx_model_base_path="ResNet18_onnx", device_name="machine_001"):
    """
    완전한 파이프라인: WAV 파일 → .pt 파일 생성 → 각 부품별 전용 ONNX 모델로 분류 분석
    
    Args:
        wav_file_path: 입력 WAV 파일 경로
        target_parts: 분석할 부품 리스트 (예: ['fan', 'pump'])
        onnx_model_base_path: 부품별 ONNX 모델들이 저장된 폴더 경로
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
        print(f"  {result['part_name']}: {status} "
              f"(확률: {result['anomaly_probability']:.3f})")
    
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

def load_metadata(metadata_path):
    """metadata.json 파일을 로드합니다."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ metadata.json 로드 실패: {e}")
        return None

def find_test_samples(test_dir):
    """test 디렉토리에서 sample 폴더들을 찾습니다."""
    sample_pattern = os.path.join(test_dir, "sample*")
    sample_dirs = glob.glob(sample_pattern)
    return sorted(sample_dirs)

def process_batch_samples(test_dir, target_parts=None, onnx_model_base_path="ResNet18_onnx"):
    """
    test 디렉토리의 모든 sample 폴더를 배치 처리합니다.
    
    Args:
        test_dir: test 디렉토리 경로
        target_parts: 분석할 부품 리스트 (None이면 metadata에서 추출)
        onnx_model_base_path: 부품별 ONNX 모델들이 저장된 폴더 경로
    
    Returns:
        list: 각 샘플의 처리 결과 리스트
    """
    
    print("🚀 배치 처리 시작")
    print("="*60)
    
    # 모델 로드 (한 번만)
    print("🔧 Demucs 모델 로딩 중...")
    model, source_names = load_model()
    init_resampler(model.samplerate)
    
    # 샘플 폴더들 찾기
    sample_dirs = find_test_samples(test_dir)
    
    if not sample_dirs:
        print(f"❌ {test_dir}에서 sample 폴더를 찾을 수 없습니다.")
        return []
    
    print(f"📁 발견된 샘플: {len(sample_dirs)}개")
    for sample_dir in sample_dirs:
        print(f"  📂 {os.path.basename(sample_dir)}")
    
    all_results = []
    
    for i, sample_dir in enumerate(sample_dirs):
        sample_name = os.path.basename(sample_dir)
        print(f"\n🔄 처리 중: {sample_name} ({i+1}/{len(sample_dirs)})")
        
        try:
            # 필수 파일 확인
            mixture_path = os.path.join(sample_dir, "mixture.wav")
            metadata_path = os.path.join(sample_dir, "metadata.json")
            
            if not os.path.exists(mixture_path):
                print(f"❌ mixture.wav 없음: {sample_dir}")
                continue
                
            if not os.path.exists(metadata_path):
                print(f"❌ metadata.json 없음: {sample_dir}")
                continue
            
            # metadata 로드
            metadata = load_metadata(metadata_path)
            if not metadata:
                continue
            
            # target_parts 결정 (metadata에서 추출하거나 사용자 지정)
            if target_parts is None and 'components' in metadata:
                current_target_parts = list(metadata['components'].keys())
            else:
                current_target_parts = target_parts or ["fan", "pump", "slider", "bearing", "gearbox"]
            
            print(f"🎯 분석 대상: {current_target_parts}")
            
            # 파이프라인 실행
            result = main_pipeline(
                wav_file_path=mixture_path,
                target_parts=current_target_parts,
                onnx_model_base_path=onnx_model_base_path,
                device_name=sample_name
            )
            
            # metadata 정보 추가
            result["metadata"] = metadata
            result["sample_info"] = {
                "sample_dir": sample_dir,
                "sample_name": sample_name
            }
            
            all_results.append(result)
            print(f"✅ {sample_name} 처리 완료")
            
        except Exception as e:
            print(f"❌ {sample_name} 처리 실패: {e}")
            error_result = {
                "sample_info": {
                    "sample_name": sample_name,
                    "sample_dir": sample_dir
                },
                "error": str(e),
                "metadata": None,
                "pipeline_info": None,
                "analysis_results": None
            }
            all_results.append(error_result)
    
    return all_results

def save_results_to_csv(batch_results, csv_filename=None):
    """
    배치 처리 결과를 CSV 파일로 저장합니다.
    
    Args:
        batch_results: process_batch_samples의 결과
        csv_filename: 저장할 CSV 파일명
    
    Returns:
        str: 저장된 CSV 파일 경로
    """
    
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"batch_analysis_results_{timestamp}.csv"
    
    # CSV 헤더 정의
    fieldnames = [
        'sample_name',
        'split',                  # test/train/val 구분
        'part_name', 
        'ground_truth',           # metadata에서 (normal/abnormal)
        'predicted',              # AI 모델 결과 (normal/abnormal)
        'prediction_probability', # AI 모델 확률
        'correct',                # 정답 여부
        'source_file',            # 원본 소스 파일 경로
        'ground_truth_rms_db',    # 원본 RMS 값
        'mixture_file_path',
        'pt_file_path',
        'processing_timestamp'
    ]
    
    csv_data = []
    
    for result in batch_results:
        if "error" in result:
            # 에러 케이스
            sample_info = result.get('sample_info', {})
            csv_data.append({
                'sample_name': sample_info.get('sample_name', 'unknown'),
                'split': 'unknown',
                'part_name': 'ERROR',
                'ground_truth': 'unknown',
                'predicted': 'error',
                'prediction_probability': 0.0,
                'correct': False,
                'source_file': '',
                'ground_truth_rms_db': 0.0,
                'mixture_file_path': sample_info.get('sample_dir', ''),
                'pt_file_path': '',
                'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            continue
        
        # 정상 처리 케이스
        sample_info = result.get('sample_info', {})
        sample_name = sample_info.get('sample_name', 'unknown')
        metadata = result.get('metadata', {})
        pipeline_info = result.get('pipeline_info', {})
        analysis_results = result.get('analysis_results', {})
        
        mixture_path = pipeline_info.get('input_wav_file', '')
        timestamp = pipeline_info.get('timestamp', '')
        generated_files = pipeline_info.get('generated_pt_files', [])
        
        # metadata에서 split 정보 추출
        split_info = metadata.get('split', 'unknown')
        
        # 각 부품별로 행 생성
        for analysis in analysis_results.get('results', []):
            part_name = analysis.get('part_name', 'unknown')
            
            # Ground truth 추출 (새로운 metadata 구조)
            component_info = metadata.get('components', {}).get(part_name, {})
            ground_truth = component_info.get('status', 'unknown')
            source_file = component_info.get('source_file', '')
            rms_db = component_info.get('rms_db', 0.0)
            
            # 예측 결과
            predicted_bool = analysis.get('anomaly_detected', False)
            predicted = 'abnormal' if predicted_bool else 'normal'
            probability = analysis.get('anomaly_probability', 0.0)
            
            # 정답 여부 계산
            # ground_truth가 'faulty'면 'abnormal'로 간주
            gt_for_compare = 'abnormal' if ground_truth == 'faulty' else ground_truth
            correct = (gt_for_compare == predicted)
            
            # 해당 부품의 .pt 파일 경로 찾기
            pt_file_path = ''
            for pt_file in generated_files:
                if part_name in pt_file:
                    pt_file_path = pt_file
                    break
            
            csv_data.append({
                'sample_name': sample_name,
                'split': split_info,
                'part_name': part_name,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'prediction_probability': round(probability, 3),
                'correct': correct,
                'source_file': source_file,
                'ground_truth_rms_db': round(rms_db, 3),
                'mixture_file_path': mixture_path,
                'pt_file_path': pt_file_path,
                'processing_timestamp': timestamp
            })
    
    # CSV 파일 저장
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n📊 CSV 결과 저장: {csv_filename}")
    print(f"📈 총 {len(csv_data)}개 행 저장")
    
    # 간단한 통계 출력
    if csv_data:
        total_predictions = len([row for row in csv_data if row['part_name'] != 'ERROR'])
        correct_predictions = len([row for row in csv_data if row['correct'] == True])
        
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"🎯 전체 정확도: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    return csv_filename

# === 메인 실행 부분 ===
if __name__ == "__main__":
    # 설정값들
    
    # 단일 파일 처리 모드
    SINGLE_MODE = False  # False로 설정하면 배치 처리 모드
    
    if SINGLE_MODE:
        # === 단일 파일 처리 ===
        WAV_FILE = "test01/mixture.wav"  # 입력 WAV 파일
        TARGET_PARTS = ["fan", "pump"]   # 분석할 부품들
        ONNX_MODEL_BASE_PATH = "ResNet18_onnx"  # 부품별 ONNX 모델들이 저장된 폴더
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
    
    else:
        # === 배치 처리 모드 ===
        TEST_DIR = "test2"  # test 디렉토리 경로
        TARGET_PARTS = None  # None이면 metadata에서 자동 추출
        ONNX_MODEL_BASE_PATH = "ResNet18_onnx"  # 부품별 ONNX 모델들이 저장된 폴더
        
        try:
            print("🚀 배치 처리 모드 시작")
            print(f"📂 테스트 디렉토리: {TEST_DIR}")
            
            # 배치 처리 실행
            batch_results = process_batch_samples(
                test_dir=TEST_DIR,
                target_parts=TARGET_PARTS,
                onnx_model_base_path=ONNX_MODEL_BASE_PATH
            )
            
            if not batch_results:
                print("❌ 처리할 샘플이 없습니다.")
                sys.exit(1)
            
            # 결과를 CSV로 저장
            csv_file = save_results_to_csv(batch_results)
            
            # 전체 배치 결과를 JSON으로도 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_json_file = f"batch_results_{timestamp}.json"
            
            with open(batch_json_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            
            print(f"📊 배치 결과 JSON 저장: {batch_json_file}")
            print(f"📈 처리 완료: {len(batch_results)}개 샘플")
            print("\n🎉 배치 처리가 성공적으로 완료되었습니다!")
            
        except Exception as e:
            print(f"❌ 배치 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
