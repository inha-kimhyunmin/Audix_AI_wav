# Audix Prepr## 🔄 처리 흐름

1. *## 🔧 잡음 제거 방식

본 시스템은 **Spectral Gating 기법을 사용하지 않습니다**. 대신 다음과 같은 방식으로 오디오를 전처리합니다:

- ❌ **Spectral Gating 미사용**: `noisereduce` 라이브러리의 spectral gating 기법 제거
- ✅ **적응적 레벨 조정 사용**: 
  - 🔊 **작은 소리 증폭**: RMS가 낮은 오디오는 최대 20dB까지 증폭
  - 🔇 **큰 소리 압축**: 임계값(0.7) 이상의 큰 소리는 3:1 비율로 소프트 압축
  - 📊 **적절한 범위**: ±3dB 내에서 미세 조정
- ✅ **Demucs 모델**: 조정된 오디오를 Demucs 모델로 소스 분리
  - **분리 소스**: `fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise` (총 6개)
  - **저장 소스**: `fan`, `pump`, `slider`, `bearing`, `gearbox` (5개, noise 제외)드** → 모노 채널, 10초, 44.1kHz로 정규화
2. **스테레오 변환** → 모델 요구사항에 맞춰 2채널로 복제
3. **적응적 레벨 조정** → 작은 소리는 증폭, 큰 소리는 압축하여 -12dB 레벨로 조정
4. **소스 분리** → Demucs 모델로 6개 소스 분리 (`fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise`)
5. **선택적 저장** → `noise`를 제외한 5개 소스만 Mel spectrogram으로 변환하여 저장
6. **JSON 결과** → 저장된 파일 정보를 JSON으로 반환

> **참고**: 모델은 6개 소스를 분리하지만, `noise`는 저장하지 않고 5개 기계 부품 소리만 저장합니다.
이 프로젝트는 **WAV 파일을 입력받아** Demucs 모델로 소리를 분리하고 Mel spectrogram으로 변환하여 저장하는 오디오 전처리 시스템입니다.

## 🎵 입력 파일 형식

- **파일 형식**: WAV 파일 (.wav)
- **길이**: 10초 고정
- **샘플링 레이트**: 44,100Hz
- **채널**: 모노 채널 (1채널)

> **참고**: 이 시스템은 실시간 녹음이 아닌 **사전에 녹음된 WAV 파일을 처리**합니다.

## � 처리 흐름

1. **WAV 파일 로드** → 모노 채널, 10초, 44.1kHz로 정규화
2. **스테레오 변환** → 모델 요구사항에 맞춰 2채널로 복제
3. **RMS 정규화** → -12dB 레벨로 볼륨 조정
4. **소스 분리** → Demucs 모델로 6개 소스 분리 (`fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise`)
5. **선택적 저장** → `noise`를 제외한 5개 소스만 Mel spectrogram으로 변환하여 저장
6. **JSON 결과** → 저장된 파일 정보를 JSON으로 반환

> **참고**: 모델은 6개 소스를 분리하지만, `noise`는 저장하지 않고 5개 기계 부품 소리만 저장합니다.

## �🔧 잡음 제거 방식

본 시스템은 **Spectral Gating 기법을 사용하지 않습니다**. 대신 다음과 같은 방식으로 오디오를 전처리합니다:

- ❌ **Spectral Gating 미사용**: `noisereduce` 라이브러리의 spectral gating 기법 제거
- ✅ **RMS 정규화 사용**: 오디오 신호를 -12dB RMS 레벨로 정규화하여 일관된 볼륨 유지
- ✅ **Demucs 모델**: 정규화된 오디오를 Demucs 모델로 소스 분리
  - **분리 소스**: `fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise` (총 6개)
  - **저장 소스**: `fan`, `pump`, `slider`, `bearing`, `gearbox` (5개, noise 제외)

## 📁 파일 라벨링 방식

처리된 결과는 다음과 같은 명명 규칙으로 저장됩니다:

```
output/시간_마이크명_부품명.pt
```

### 파일명 구성 요소:
- **시간**: `YYYY-MM-DD_HH-MM-SS` 형식 (예: `2025-07-24_15-30-45`)
- **마이크명**: `mic_1`, `mic_2` 형식
- **부품명**: `fan`, `pump`, `slider`, `bearing`, `gearbox` (noise는 저장하지 않음)

### 예시 파일명:
```
output/2025-07-24_15-30-45_mic_1_fan.pt
output/2025-07-24_15-30-45_mic_1_pump.pt
output/2025-07-24_15-30-45_mic_2_bearing.pt
output/2025-07-24_15-30-45_mic_2_gearbox.pt
```

## 📊 JSON 결과 생성 함수

시스템은 처리 완료 후 구조화된 JSON 결과를 반환합니다.

### 단일 파일 처리: `process_wav_file()`

```python
def process_wav_file(model, source_names, wav_path):
    """
    WAV 파일을 처리하고 결과를 JSON으로 반환합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트
    :param wav_path: 입력 WAV 파일 경로
    :return: 처리 결과 JSON
    """
```

**반환 JSON 구조:**
```json
{
  "input_file": "test01/mix.wav",
  "timestamp": "2025-07-24_15-30-45",
  "processed_files": [
    {
      "mic_number": 1,
      "part_name": "fan",
      "file_path": "output/2025-07-24_15-30-45_mic_1_fan.pt"
    },
    {
      "mic_number": 1,
      "part_name": "pump", 
      "file_path": "output/2025-07-24_15-30-45_mic_1_pump.pt"
    }
  ]
}
```

### 배치 처리: `process_multiple_wav_files()`

```python
def process_multiple_wav_files(model, source_names, wav_paths):
    """
    여러 WAV 파일을 배치로 처리합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트
    :param wav_paths: WAV 파일 경로 리스트
    :return: 모든 처리 결과를 포함한 JSON
    """
```

**반환 JSON 구조:**
```json
{
  "batch_processing": true,
  "total_files": 3,
  "successful_files": 2,
  "results": [
    {
      "input_file": "file1.wav",
      "timestamp": "2025-07-24_15-30-45",
      "processed_files": [...]
    },
    {
      "input_file": "file2.wav",
      "error": "File not found",
      "processed_files": []
    }
  ]
}
```

### JSON 필드 설명:
- **`mic_number`**: 마이크 번호 (1, 2, ...)
- **`part_name`**: 분리된 부품명 (`fan`, `pump`, `slider`, `bearing`, `gearbox`) - noise 제외
- **`file_path`**: 저장된 .pt 파일의 전체 경로
- **`timestamp`**: 처리 시점의 타임스탬프
- **`input_file`**: 입력 WAV 파일 경로

## 주요 기능

1. **WAV 파일 입력**: 10초 길이, 44.1kHz, 모노 채널 WAV 파일 처리
2. **RMS 정규화**: 입력 오디오를 -12dB RMS 레벨로 정규화
3. **소리 분리**: Demucs 모델을 사용한 소스 분리
4. **Mel Spectrogram 변환**: 분리된 오디오를 Mel spectrogram으로 변환
5. **JSON 결과 출력**: 처리 결과를 JSON 형식으로 반환

## 변경사항 (v2.0)

- ❌ **Spectral Gating 잡음 제거 제거**: noisereduce 라이브러리를 사용한 잡음 제거 기능 제거
- ✅ **RMS 정규화 추가**: 오디오를 -12dB RMS 레벨로 정규화하여 일관된 볼륨 유지
- ✅ **WAV 파일 입력**: 실시간 녹음 대신 WAV 파일을 입력받아 처리
- ✅ **파일명 형식 변경**: `시간_마이크명_부품명.pt` 형식으로 출력 파일 저장
- ✅ **JSON 결과 반환**: 처리 결과를 구조화된 JSON으로 반환

## 설정 (config.py)

```python
# 적응적 레벨 조정 설정
TARGET_RMS_DB = -12.0        # 목표 RMS 레벨 (dB) - 수정 가능
MAX_GAIN_DB = 20.0           # 최대 증폭 게인 (dB) - 수정 가능
COMPRESSION_THRESHOLD = 0.7  # 압축 시작 임계값 (0~1) - 수정 가능
RMS_EPSILON = 1e-9          # 0으로 나누기 방지용 작은 값
```

### 설정 매개변수 설명:
- **TARGET_RMS_DB**: 최종 목표하는 RMS 레벨
- **MAX_GAIN_DB**: 작은 소리 증폭 시 최대 게인 제한
- **COMPRESSION_THRESHOLD**: 이 값 이상의 크기를 가진 신호는 압축 적용

기본 라이브러리 

- numpy 배열 사용을 위한 numpy
- PyTorch 사용을 위한 torch
- PyTorch 형식으로 오디오 처리를 위한 torchaudio (mel spectrogram 변환에 사용)

## 1. sounddevice 라이브러리에서 입력이 들어오는 구조

오디오 인터페이스(여러 개의 마이크 입력을 받는 장치)에서는 아날로그 마이크 신호를 디지털 데이터로 변환한다. 이를 지속적으로 스트림으로 넘겨주는 형태

간단하게 하자면 배열에서의 인덱스 값이 주기적으로 계속해서 넘어온다.

그러면 sounddevice는 오디오 입력의 채널 수(마이크 개수), 샘플링 레이트(Hz), 샘플 길이(sampling rate * time) 에 맞춰 버퍼 할당 후

각 버퍼로 입력들이 들어온다.

[디지털 변환 과정 세부 설명](https://www.notion.so/2328f8ab094e80b198aadeb260e7300b?pvs=21)

record.py

```python
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, SEGMENT_DURATION

def record_segment():
    """
    오디오 세그먼트를 녹음합니다.
    :return: 녹음된 오디오 데이터 (numpy 배열)
    """
    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']  # type: ignore
    print(f"🎙 녹음 시작 (채널: {channels})")
    audio = sd.rec(int(SAMPLE_RATE * SEGMENT_DURATION), samplerate=SAMPLE_RATE,
                   channels=channels, dtype='float32')
    sd.wait()
    return audio  # [samples, channels]

```

여기 함수에서는 리턴값은 numpy 2차원 배열 ([[1,2,3,4,5],[6,8,4,1,2],[1,8,9,0,1]] 하나의 배열이 채널 하나 총 3개의 배열 → 3채널)

오디오 입력을 받고, for문을 돌려서 뒤의 과정은 채널별로 각각 수행

반복 횟수는 채널 개수만큼

main.py

```python
from config import NOISE_SAMPLE_PATH
from record import record_segment
from denoise import load_noise_clip, denoise
from model import load_model, separate
from mel import save_mel_tensor

def process_stream(model, repeat=5):
    """
    스트림을 처리하고 오디오를 녹음, 잡음 제거, 분리 및 저장합니다.
    
    :param model: 분리 모델
    :param repeat: 반복 횟수
    :return: None
    """
    noise_clip = load_noise_clip()
    for idx in range(repeat):
        audio = record_segment()
        for mic_idx in range(audio.shape[1]): #여기서 각 채널별로 전처리 진행
            print(f"\n🎧 마이크 {mic_idx} 처리 중")
            clean = denoise(audio[:, mic_idx], noise_clip)
            sources = separate(model, clean)
            for src_idx, src in enumerate(sources):
                save_mel_tensor(src, mic_idx, src_idx, idx)

if __name__ == "__main__":
    model = load_model()
    process_stream(model, repeat=100) # 10초 * 100회 = 1000초 (약 16분)

```

## 2. 전처리 - 1 Spectral Gating

환경 잡음 제거 - Spectral Gating  기법 사용 - noisereduce 라이브러리 사용

환경 잡음(noise sample)을 추출하기 위해 기계 소리 녹음 전 환경 소리를 녹음하는 과정이 필요하다. 이를 noise sample로 저장(.pt파일로)

### 입력 :  1차원 numpy 배열

각 채널마다의 numpy배열을 주파수 스펙트럼으로 변환(STFT 기법이 있음)

주파수 스펙트럼에서 noise sample보다 작은 소리들을 전부 제거

이를 다시 시간 영역으로 변환 

[Spectral Gating 세부 설명](https://www.notion.so/Spectral-Gating-2328f8ab094e805ab62ef4484e04eb0e?pvs=21)

### 출력 : 1차원 numpy 배열

## 3. 소리 분리 모델 - demucs

demucs 라이브러리 사용

**모델 분리 능력:**
- **입력**: mixture.wav (혼합 오디오)
- **출력**: 6개 소스 (`fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise`)
- **저장**: 5개 소스 (기계 부품만, `noise`는 제외)

pre-trained 모델을 사용할 때에는 그냥 demucs 라이브러리만 설치하면 되는데

따로 훈련한 모델을 사용할 때는 어떤 라이브러리를 추가로 설치해야하는지 모르겠음. 실제로 돌려봐야 알거같음

### 입력 : 2차원 numpy 배열

훈련한 모델을 불러와서 소스 분리 수행

### 출력 : tensor 파일 (noise 제외하고 5개)

## 4. Mel Spectrogram 변환 → tensor 파일 변환

tensor 파일을 변환한 후 mel spectrogram으로 변환하고, 이를 tensor 파일로 변환함

tensor 파일은 종명이의 요구인 1,240,240 크기로 지정

### 입력 : tensor 파일(하나의 오디오 채널의 정보)

### 출력 : tensor 파일(.pt)

출력은 각 채널별로, 그리고 각 부품의 소리로 분리되어 .pt가 저장된다

## 💻 사용 예시

### 단일 WAV 파일 처리:
```python
from main import process_wav_file
from model import load_model

# 모델 로드
model, source_names = load_model()

# WAV 파일 처리
result = process_wav_file(model, source_names, "input_audio.wav")

# 결과 출력
print(result)
```

### 배치 처리:
```python
from main import process_multiple_wav_files

# 여러 파일 동시 처리
wav_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = process_multiple_wav_files(model, source_names, wav_files)

print(f"처리 완료: {results['successful_files']}/{results['total_files']} 파일")
```

### 커맨드 라인 실행:
```bash
python main.py
```

## 📂 출력 폴더 구조

**기존 방식 (v1.0):**
```
output/
└── 2025-07-16_15-03-20/   # 측정 시간 폴더
    ├── mic_1/
    │   ├── fan.pt
    │   ├── pump.pt
    │   └── ...
    ├── mic_2/
    │   └── ...
```

**새로운 방식 (v2.0):**
```
output/
├── 2025-07-24_15-30-45_mic_1_fan.pt
├── 2025-07-24_15-30-45_mic_1_pump.pt
├── 2025-07-24_15-30-45_mic_2_bearing.pt
├── 2025-07-24_15-30-45_mic_2_gearbox.pt
└── 2025-07-24_15-30-45_mic_2_slider.pt
```

> **개선점**: 파일명에 모든 정보가 포함되어 파일 관리가 용이하고, 다음 처리 단계에서 파일 경로를 쉽게 파악할 수 있습니다.