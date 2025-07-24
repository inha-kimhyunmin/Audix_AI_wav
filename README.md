# Audix Preprocessing System

이 프로젝트는 **WAV 파일을 입력받아** Demucs 모델로 소리를 분리하고 Mel spectrogram으로 변환하여 저장하는 오디오 전처리 시스템입니다.

## 🎵 입력 파일 형식

- **파일 형식**: WAV 파일 (.wav)
- **길이**: 10초 고정
- **샘플링 레이트**: 44,100Hz
- **채널**: 모노 채널 (1채널)

> **참고**: 이 시스템은 실시간 녹음이 아닌 **사전에 녹음된 WAV 파일을 처리**합니다.

## 🔄 처리 흐름

1. **WAV 파일 로드** → 모노 채널, 10초, 44.1kHz로 정규화
2. **스테레오 변환** → 모델 요구사항에 맞춰 2채널로 복제
3. **적응적 레벨 조정** → 작은 소리는 증폭, 큰 소리는 압축하여 -12dB 레벨로 조정
4. **소스 분리** → Demucs 모델로 6개 소스 분리 (`fan`, `pump`, `slider`, `bearing`, `gearbox`, `noise`)
5. **선택적 저장** → `noise`를 제외한 5개 소스만 Mel spectrogram으로 변환하여 저장
6. **JSON 결과** → 저장된 파일 정보를 JSON으로 반환

> **참고**: 모델은 6개 소스를 분리하지만, `noise`는 저장하지 않고 5개 기계 부품 소리만 저장합니다.

## 🔧 잡음 제거 방식

본 시스템은 **Spectral Gating 기법을 사용하지 않습니다**. 대신 다음과 같은 방식으로 오디오를 전처리합니다:

- ❌ **Spectral Gating 미사용**: `noisereduce` 라이브러리의 spectral gating 기법 제거
- ✅ **적응적 레벨 조정 사용**: 
  - 🔊 **작은 소리 증폭**: RMS가 낮은 오디오는 최대 20dB까지 증폭
  - 🔇 **큰 소리 압축**: 임계값(0.7) 이상의 큰 소리는 3:1 비율로 소프트 압축
  - 📊 **적절한 범위**: ±3dB 내에서 미세 조정
- ✅ **Demucs 모델**: 조정된 오디오를 Demucs 모델로 소스 분리
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
2. **적응적 레벨 조정**: 작은 소리 증폭 및 큰 소리 압축으로 일관된 볼륨 유지
3. **소리 분리**: Demucs 모델을 사용한 6개 소스 분리 (5개 저장)
4. **Mel Spectrogram 변환**: 분리된 오디오를 240x240 mel spectrogram으로 변환
5. **ONNX 이상 감지**: 각 부품별 이상 감지 수행
6. **JSON 결과 출력**: 처리 결과 및 이상 감지 결과를 JSON 형식으로 반환

## 변경사항 (v3.0)

- ❌ **Spectral Gating 제거**: noisereduce 라이브러리를 사용한 잡음 제거 기능 완전 제거
- ✅ **적응적 레벨 조정 도입**: RMS 기반 단순 정규화를 적응적 레벨 조정으로 대체
  - 작은 소리 증폭 (최대 20dB)
  - 큰 소리 압축 (3:1 비율)
  - 미세 조정 (±3dB)
- ✅ **ONNX 모델 연동**: Mel spectrogram을 ONNX 모델에 입력하여 이상 감지 수행
- ✅ **통합 분석 시스템**: WAV → 분리 → 분류의 완전한 파이프라인 구축
- ✅ **향상된 JSON 출력**: 이상 감지 결과 및 확률값 포함

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

## 🤖 ONNX 모델 연동

### ONNX 모델 개요

분리된 mel spectrogram (.pt 파일)을 ONNX 모델에 입력하여 이상 감지를 수행합니다.

### 입력 형식

**파일**: `.pt` 파일 (mel spectrogram)
- **크기**: `(1, 240, 240)` 또는 `(240, 240)`
- **타입**: `float32`
- **형태**: Mel spectrogram tensor

### ONNX 모델 처리 과정

```python
def predict_single_file_onnx_json(onnx_model_path, pt_file_path, device_name, in_ch=1, threshold=0.5):
    """
    단일 .pt 파일에 대해 ONNX 모델로 이상 감지를 수행합니다.
    
    Args:
        onnx_model_path: ONNX 모델 파일 경로
        pt_file_path: 입력 .pt 파일 경로 (mel spectrogram)
        device_name: 장치명
        in_ch: 입력 채널 수 (기본값: 1)
        threshold: 이상 감지 임계값 (기본값: 0.5)
    
    Returns:
        dict: JSON 형태의 예측 결과
    """
```

#### 1. **전처리 단계**
```python
# .pt 파일 로드
x = torch.load(pt_file_path).float()

# 차원 조정
if x.ndim == 2:
    x = x.unsqueeze(0)  # [H, W] → [1, H, W]

# 채널 수 맞추기
if x.shape[0] != in_ch:
    x = x.repeat(in_ch, 1, 1)

# 배치 차원 추가 및 numpy 변환
x = x.unsqueeze(0).numpy().astype(np.float32)  # [1, C, H, W]
```

#### 2. **ONNX 추론**
```python
# ONNX 세션 생성
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# 추론 실행
outputs = session.run([output_name], {input_name: x})
logit = outputs[0][0][0]  # scalar 값

# Sigmoid 활성화 함수 적용
prob = float(1 / (1 + np.exp(-logit)))
```

### 출력 형식

**JSON 구조:**
```json
{
  "device_name": "machine_001",
  "result": true,
  "probability": 0.942
}
```

**필드 설명:**
- **`device_name`**: 장치명 (사용자 입력)
- **`result`**: 이상 감지 결과 (boolean)
  - `true`: 이상 감지됨 (probability ≥ threshold)
  - `false`: 정상 (probability < threshold)
- **`probability`**: 이상일 확률 (0.0 ~ 1.0)

### 통합 분석 시스템

**전체 파이프라인**: `integrated_analysis.py`

```python
def process_audio_with_classification(wav_file_path, onnx_model_path, mic_number, device_name):
    """
    WAV 파일 → Demucs 분리 → ONNX 분류의 완전한 파이프라인
    """
```

#### 실행 예시:
```python
# 설정
WAV_FILE = "test01/mixture.wav"
ONNX_MODEL = "ResNet18_onnx/fold0_best_model.onnx"
MIC_NUMBER = 1
DEVICE_NAME = "machine_001"

# 통합 처리 실행
results = process_audio_with_classification(
    wav_file_path=WAV_FILE,
    onnx_model_path=ONNX_MODEL,
    mic_number=MIC_NUMBER,
    device_name=DEVICE_NAME
)
```

#### 통합 결과 JSON:
```json
{
  "input_wav_file": "test01/mixture.wav",
  "mic_number": 1,
  "device_name": "machine_001",
  "total_parts": 5,
  "anomaly_count": 3,
  "results": [
    {
      "mic_number": 1,
      "part_name": "fan",
      "pt_file_path": "output/2025-07-24_15-30-45_mic_1_fan.pt",
      "device_name": "machine_001",
      "anomaly_detected": true,
      "anomaly_probability": 0.942
    },
    {
      "mic_number": 1,
      "part_name": "pump", 
      "pt_file_path": "output/2025-07-24_15-30-45_mic_1_pump.pt",
      "device_name": "machine_001",
      "anomaly_detected": false,
      "anomaly_probability": 0.234
    }
  ]
}
```

### ONNX 모델 요구사항

- **런타임**: ONNX Runtime (`pip install onnxruntime`)
- **실행 환경**: CPU (CPUExecutionProvider)
- **입력 형태**: `[1, 1, 240, 240]` (batch, channel, height, width)
- **출력 형태**: `[1, 1]` (batch, logit)
- **모델 타입**: 이진 분류 모델 (정상/이상)

## 💻 사용 예시

### 1. 단일 WAV 파일 처리 (기본)
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

### 2. 통합 분석 (Demucs + ONNX)
```python
from integrated_analysis import process_audio_with_classification

# 통합 처리 실행
results = process_audio_with_classification(
    wav_file_path="test01/mixture.wav",
    onnx_model_path="ResNet18_onnx/fold0_best_model.onnx", 
    mic_number=1,
    device_name="machine_001"
)

# 결과 확인
print(f"이상 감지된 부품: {results['anomaly_count']}/{results['total_parts']}")
for result in results['results']:
    status = "🚨 이상" if result['anomaly_detected'] else "✅ 정상"
    print(f"{result['part_name']}: {status} (확률: {result['anomaly_probability']:.3f})")
```

### 3. 배치 처리
```python
from main import process_multiple_wav_files

# 여러 파일 동시 처리
wav_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = process_multiple_wav_files(model, source_names, wav_files)

print(f"처리 완료: {results['successful_files']}/{results['total_files']} 파일")
```

### 4. 커맨드 라인 실행
```bash
# 기본 처리
python main.py

# 통합 분석
python integrated_analysis.py

# 평가 (선택사항)
python seperate_evaluate.py
```

## 📂 출력 폴더 구조

```
output/
├── 2025-07-24_15-30-45_mic_1_fan.pt
├── 2025-07-24_15-30-45_mic_1_pump.pt
├── 2025-07-24_15-30-45_mic_2_bearing.pt
├── 2025-07-24_15-30-45_mic_2_gearbox.pt
└── 2025-07-24_15-30-45_mic_2_slider.pt
```

**파일명 형식**: `YYYY-MM-DD_HH-MM-SS_mic_N_PART.pt`
- 시간 정보, 마이크 번호, 부품명이 모두 포함되어 관리 용이

## 📊 시스템 아키텍처

### 전체 워크플로우

```
📁 WAV 파일 (10초, 44.1kHz, mono)
    ↓
🔧 적응적 레벨 조정 (Adaptive Level Adjustment)
    ↓ 
🎵 Demucs 소스 분리 (6개 소스 → 5개 저장)
    ↓
🖼️ Mel Spectrogram 변환 (.pt 파일들)
    ↓
🤖 ONNX 이상 감지 모델
    ↓
📊 JSON 결과 (정상/이상 + 확률)
```

### 주요 컴포넌트

1. **전처리 모듈** (`main.py`, `rms_normalize.py`)
   - WAV 파일 로드 및 정규화
   - 적응적 레벨 조정

2. **소스 분리 모듈** (`model.py`)
   - Demucs 모델 로드 및 추론
   - 6개 소스 분리 (noise 제외하고 5개 저장)

3. **Mel Spectrogram 변환** (`mel.py`)
   - 오디오를 240x240 mel spectrogram으로 변환
   - .pt 파일로 저장

4. **ONNX 분류 모듈** (`onnx.py`)
   - .pt 파일을 ONNX 모델에 입력
   - 이상 감지 수행 및 JSON 결과 반환

5. **통합 시스템** (`integrated_analysis.py`)
   - 전체 파이프라인 통합 관리
   - 배치 처리 및 결과 취합

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

---

# 🔄 Version History & Changes

## v2.0 - Major Architecture Refactoring (Latest)

### 📋 주요 변경사항 (First Commit 이후)

#### 🏗️ 아키텍처 개선
- **역할 분리**: 단일 통합 시스템을 3개 모듈로 분리
  - `audio_preprocessing.py`: .pt 파일 생성 전용
  - `integrated_analysis.py`: ONNX 분류 분석 전용  
  - `main.py`: 통합 파이프라인 실행
- **단순화**: 복잡한 멀티 마이크 처리 로직 제거, 단일 WAV 파일 처리로 단순화
- **타겟 부품 지정**: 사용자가 분석할 부품을 선택적으로 지정 가능

#### 📁 파일 구조 변화
**삭제된 파일들:**
- `convert_folder_to_mel.py` - 폴더 배치 처리 로직 main.py로 통합
- `monotostereo.py` - 스테레오 변환 로직 간소화
- `noise_profile.py` - Spectral Gating 관련 기능 제거
- `noise_sample.pt` - 노이즈 프로파일 샘플 제거
- `record.py` - 실시간 녹음 기능 제거 (WAV 파일 처리로 대체)
- `seperate_evaluate_new.py` - 평가 로직 정리

**새로 추가된 파일들:**
- `audio_preprocessing.py` - 전처리 전용 모듈 (기존 main.py에서 분리)

#### 🎯 기능 개선
- **선택적 부품 처리**: `target_parts` 매개변수로 필요한 부품만 처리
- **효율성 향상**: 불필요한 .pt 파일 생성 방지
- **모듈화**: 각 단계별 독립 실행 가능
- **JSON 출력**: 구조화된 분석 결과 제공

#### 🔧 설정 파일 업데이트
- `config.py`: 새로운 아키텍처에 맞춰 설정 간소화
- 멀티 채널 관련 설정 제거
- 타겟 부품 지정 시스템 도입

### 🚀 사용법 변화

**v1.0 (First Commit):**
```python
# 복잡한 멀티 마이크 처리
python main.py  # 모든 마이크, 모든 부품 처리
```

**v2.0 (Latest):**
```python
# 1. .pt 파일만 생성
python audio_preprocessing.py

# 2. 분석만 수행  
python integrated_analysis.py

# 3. 전체 파이프라인
python main.py
```

### 📊 성능 개선
- **처리 시간 단축**: 불필요한 부품 처리 건너뛰기
- **메모리 효율성**: 선택적 부품 처리로 메모리 사용량 감소
- **파일 관리**: 명확한 파일명 규칙으로 결과 추적 용이

### 🎯 Target Parts 시스템
```python
# 원하는 부품만 선택적으로 처리
target_parts = ["fan", "pump"]  # gearbox, slider, bearing 건너뛰기

# 처리 결과
✅ fan.pt 생성
✅ pump.pt 생성  
⏭️ slider 건너뛰기
⏭️ bearing 건너뛰기
⏭️ gearbox 건너뛰기
```

### 🔄 Migration Guide

**기존 사용자 (v1.0):**
1. `python main.py` → `python main.py` (동일)
2. 단, `target_parts` 매개변수로 부품 선택 가능

**새로운 사용자:**
1. 단계별 실행: `audio_preprocessing.py` → `integrated_analysis.py`
2. 통합 실행: `main.py`

---