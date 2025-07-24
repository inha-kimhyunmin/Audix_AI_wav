import os
import torchaudio
from datetime import datetime
from mel import save_mel_tensor
from config import SAMPLE_RATE, OUTPUT_FOLDER

# === 사용자 설정 ===
INPUT_FOLDER = "D:/machine_sounds"
MIC_IDX = 0
TIMESTAMP_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def convert_all_audio_recursively(input_folder):
    folder_counts = {}  # 각 폴더 이름별 카운터

    for root, _, files in os.walk(input_folder):
        files = sorted([f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))])
        if not files:
            continue

        # 폴더 이름 얻기
        folder_name = os.path.basename(root)
        if folder_name == os.path.basename(input_folder):
            # 최상위 폴더면 스킵 (파일이 있을 경우에만 해당됨)
            continue

        folder_counts.setdefault(folder_name, 1)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                waveform, sr = torchaudio.load(file_path)

                # 모노 처리
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 리샘플링
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    waveform = resampler(waveform)

                # 파일 이름: foldername001.pt, foldername002.pt, ...
                count = folder_counts[folder_name]
                numbered_name = f"{folder_name}{count:03d}"
                folder_counts[folder_name] += 1

                save_mel_tensor(
                    source_tensor=waveform,
                    mic_idx=MIC_IDX,
                    source_name=numbered_name,
                    timestamp_str=TIMESTAMP_STR,
                    parts_to_save=None
                )

                print(f"✅ Processed: {file_path} → {numbered_name}")

            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    convert_all_audio_recursively(INPUT_FOLDER)
