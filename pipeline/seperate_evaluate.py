import os
import torchaudio
import torch
import numpy as np
from .model import load_model, separate  # 너의 Demucs 로딩 함수 사용
from rms_normalize import adaptive_level_adjust  # 적응적 레벨 조정 사용 (main.py와 동일)
from config import SOURCES as CONFIG_SOURCES  # config.py의 SOURCES 사용
from pathlib import Path

# === 사용자 설정 ===
FOLDER = "C:/Users/dotor/Desktop/Audix_Preprocessing/test01"
# config.py와 일치하는 SOURCES 사용하되, noise는 평가에서 제외
EVAL_SOURCES = [src for src in CONFIG_SOURCES if src.lower() != 'noise']
print(f"평가할 소스들: {EVAL_SOURCES}")
print(f"모델 전체 소스들: {CONFIG_SOURCES}")

# === SI-SDR 계산 함수 ===
def compute_sisdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = torch.sum(est * ref) / torch.sum(ref ** 2)
    proj = alpha * ref
    noise = est - proj
    sisdr = 10 * torch.log10(torch.sum(proj ** 2) / torch.sum(noise ** 2))
    return sisdr.item()

# === 분리 및 평가 ===
def evaluate_demucs_on_folder(folder):
    """
    Demucs 모델의 소스 분리 성능을 평가합니다.
    분리된 소스와 Ground Truth 모두에 적응적 레벨 조정을 적용하여 평가합니다.
    
    :param folder: 평가 데이터가 있는 폴더 경로
    :return: 소스별 SI-SDR 점수 딕셔너리
    """
    # 0. 폴더와 필수 파일 존재 확인
    if not os.path.exists(folder):
        raise FileNotFoundError(f"평가 폴더가 존재하지 않습니다: {folder}")
    
    mix_path = os.path.join(folder, "mixture.wav")
    if not os.path.exists(mix_path):
        raise FileNotFoundError(f"mixture.wav 파일이 존재하지 않습니다: {mix_path}")
    
    print(f"📂 평가 폴더: {folder}")
    print(f"📂 mixture.wav: {mix_path}")
    
    # 1. 혼합 소리 로드
    mix_path = os.path.join(folder, "mixture.wav")
    mixture, sr = torchaudio.load(mix_path)
    if mixture.shape[0] > 1:
        mixture = mixture.mean(dim=0, keepdim=True)
    
    # 적응적 레벨 조정 적용 (main.py와 동일한 전처리)
    mixture_np = mixture.squeeze().numpy()
    mixture_np_normalized = adaptive_level_adjust(mixture_np)
    
    # adaptive_level_adjust가 tensor를 반환할 수 있으므로 numpy로 변환
    if isinstance(mixture_np_normalized, torch.Tensor):
        mixture_np_normalized = mixture_np_normalized.numpy()
    
    print(f"🔧 적응적 레벨 조정 적용 완료")
    print(f"정규화 후 타입: {type(mixture_np_normalized)}, 형태: {mixture_np_normalized.shape}")
    
    # 2. 모델 로드 및 분리
    model, model_sources = load_model()
    print(f"모델이 출력하는 소스: {model_sources}")
    
    model.eval()
    with torch.no_grad():
        print(f"mixture_np_normalized shape: {mixture_np_normalized.shape}, type: {type(mixture_np_normalized)}")
        estimates = separate(model, mixture_np_normalized)  # 정규화된 numpy 배열 전달
        print(f"estimates shape before squeeze: {estimates.shape}")
        estimates = estimates.squeeze(0)  # batch 차원 제거
        print(f"estimates shape after squeeze(0): {estimates.shape}")
        
        # estimates가 (sources, channels, samples) 형태라면 mono로 변환
        if len(estimates.shape) == 3:
            estimates = estimates.mean(dim=1)  # 채널 차원 평균내어 mono로
            print(f"estimates shape after mean: {estimates.shape}")
        elif len(estimates.shape) == 2:
            pass  # 이미 (sources, samples) 형태
        print(f"Final estimates shape: {estimates.shape}")

    # 3. 평가 (noise 제외)
    scores = {}
    for source in EVAL_SOURCES:
        # 모델 출력에서 해당 소스의 인덱스 찾기
        if source in model_sources:
            source_idx = model_sources.index(source)
        else:
            print(f"❌ {source}가 모델 출력에 없습니다. 사용 가능한 소스: {model_sources}")
            continue
            
        gt_path = os.path.join(folder, f"{source}.wav")
        if not os.path.exists(gt_path):
            print(f"❌ {source}.wav not found, skipping...")
            continue

        target, sr2 = torchaudio.load(gt_path)
        if target.shape[0] > 1:
            target = target.mean(dim=0, keepdim=True)

        # 분리된 소스 가져오기
        est = estimates[source_idx]  # 모델 출력에서 올바른 인덱스 사용
        target = target.squeeze()  # (1, samples) -> (samples)
        
        # Ground Truth와 분리된 소스 모두에 적응적 레벨 조정 적용
        target_np = target.numpy() if isinstance(target, torch.Tensor) else target
        est_np = est.numpy() if isinstance(est, torch.Tensor) else est
        
        print(f"📊 {source} Ground Truth 원본 RMS: {20 * np.log10(np.sqrt(np.mean(target_np ** 2)) + 1e-8):.2f}dB")
        print(f"📊 {source} 분리된 소스 원본 RMS: {20 * np.log10(np.sqrt(np.mean(est_np ** 2)) + 1e-8):.2f}dB")
        
        target_normalized = adaptive_level_adjust(target_np)
        est_normalized = adaptive_level_adjust(est_np)
        
        # 타입 변환 확인
        if isinstance(target_normalized, torch.Tensor):
            target_normalized = target_normalized.numpy()
        if isinstance(est_normalized, torch.Tensor):
            est_normalized = est_normalized.numpy()
        
        print(f"🔧 {source} 적응적 레벨 조정 완료 (Ground Truth & 분리된 소스)")
        
        # 길이 맞추기
        min_length = min(est_normalized.shape[0], target_normalized.shape[0])
        est_final = torch.from_numpy(est_normalized[:min_length]).unsqueeze(0)
        target_final = torch.from_numpy(target_normalized[:min_length]).unsqueeze(0)

        sisdr = compute_sisdr(est_final, target_final)
        scores[source] = sisdr
        print(f"✅ {source} (인덱스 {source_idx}): SI-SDR = {sisdr:.2f} dB")

    return scores

# === 실행 ===
if __name__ == "__main__":
    try:
        print("🚀 Demucs 모델 평가 시작")
        print(f"📁 평가 폴더: {FOLDER}")
        print(f"📊 평가 대상 소스: {EVAL_SOURCES}")
        
        # 필요한 파일들 존재 확인
        required_files = ["mixture.wav"] + [f"{src}.wav" for src in EVAL_SOURCES]
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(FOLDER, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️ 누락된 파일들: {missing_files}")
            print("평가를 위해 다음 파일들이 필요합니다:")
            for file in required_files:
                print(f"  - {file}")
        else:
            # 적응적 레벨 조정 적용 평가
            print("\n" + "="*60)
            print("🔊 적응적 레벨 조정 적용 평가 (Ground Truth와 분리된 소스 모두에 적용)")
            print("="*60)
            scores = evaluate_demucs_on_folder(FOLDER)
            
            print("\n📊 평가 결과:")
            total_score = 0
            valid_scores = 0
            
            for src, score in scores.items():
                print(f"{src}: {score:.2f} dB")
                total_score += score
                valid_scores += 1
            
            if valid_scores > 0:
                avg_score = total_score / valid_scores
                print(f"🎯 평균 SI-SDR: {avg_score:.2f} dB")
                
                # 성능 평가
                if avg_score >= 10:
                    print("🎉 우수한 분리 성능!")
                elif avg_score >= 5:
                    print("👍 양호한 분리 성능")
                elif avg_score >= 0:
                    print("📈 기본적인 분리 성능")
                else:
                    print("⚠️ 개선이 필요한 성능")
            else:
                print("❌ 평가할 수 있는 소스가 없습니다.")
            
    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("💡 해결 방법: pip install torch torchaudio 실행")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
