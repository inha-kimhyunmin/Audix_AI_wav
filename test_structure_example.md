# test 폴더 구조 예시
mkdir -p test/sample000001
mkdir -p test/sample000002

# sample000001/metadata.json 예시
{
  "sample_id": "sample000001",
  "parts": {
    "fan": "normal",
    "pump": "abnormal", 
    "slider": "normal",
    "bearing": "abnormal",
    "gearbox": "normal"
  },
  "description": "Test sample with pump and bearing anomalies"
}

# sample000002/metadata.json 예시
{
  "sample_id": "sample000002", 
  "parts": {
    "fan": "abnormal",
    "pump": "normal",
    "slider": "abnormal", 
    "bearing": "normal",
    "gearbox": "abnormal"
  },
  "description": "Test sample with fan, slider and gearbox anomalies"
}
