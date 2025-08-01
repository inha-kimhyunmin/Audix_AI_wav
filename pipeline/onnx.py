import torch
import numpy as np
import onnxruntime as ort
from datetime import datetime
import re
import os

# def extract_datetime_from_filename(filename):
#     """
#     예시: fan_normal_2025-07-24_16-21-03.pt → '2025-07-24 16:21:03'
#     """
#     match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
#     if match:
#         date_str, time_str = match.groups()
#         return date_str + " " + time_str.replace("-", ":")
#     else:
#         # fallback: 현재 시간
#         return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def predict_single_file_onnx_json(onnx_model_path, pt_file_path, device_name="unknown_device", in_ch=1, threshold=0.5):
    # 1. ONNX 세션 생성
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. .pt 파일 로드 및 전처리
    x = torch.load(pt_file_path).float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] != in_ch:
        x = x.repeat(in_ch, 1, 1)

    x = x.unsqueeze(0).numpy().astype(np.float32)  # [1, C, H, W]

    # 3. ONNX 추론
    outputs = session.run([output_name], {input_name: x})
    logit = outputs[0][0][0]  # scalar
    prob = float(1 / (1 + np.exp(-logit)))  # sigmoid

    # 4. 파일명에서 datetime 추출
    filename = os.path.basename(pt_file_path)
    # created_at = extract_datetime_from_filename(filename)

    # 5. 결과 JSON 구성
    result_json = {
        "device_name": device_name,
        "result": prob >= threshold,
        "probability": round(prob, 3)
        # "created_at": created_at
    }

    return result_json
