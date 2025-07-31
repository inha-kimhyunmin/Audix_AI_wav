import torch

SAMPLE_RATE = 44100
MEL_SAMPLE_RATE = 16000
SEGMENT_DURATION = 10
MEL_SIZE = (240, 240)
MODEL_PATH = "model/6a76e118.th"
NOISE_SAMPLE_PATH = "noise_sample.pt"
OUTPUT_FOLDER = "output"
#DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCES = ["fan", "pump", "slider", "bearing", "gearbox", "noise"]  # 모델에 따라 조정 (noise 추가)
FORCE_STEREO_INPUT = True  # 모델이 2채널 입력을 요구함

# RMS 정규화 설정
TARGET_RMS_DB = -12.0  # 목표 RMS 레벨 (dB)
RMS_EPSILON = 1e-9     # 0으로 나누기 방지용 작은 값

# 적응적 레벨 조정 설정
MAX_GAIN_DB = 20.0           # 최대 증폭 게인 (dB)
COMPRESSION_THRESHOLD = 0.7  # 압축 시작 임계값 (0~1)