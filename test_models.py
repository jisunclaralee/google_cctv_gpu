
# 모델 로드 테스트
import os
from pathlib import Path

weights_dir = Path("models/weights")
onnx_files = list(weights_dir.glob("*.onnx"))

print("📋 발견된 ONNX 모델 파일:")
for file in onnx_files:
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"✅ {file.name}: {size_mb:.1f}MB")

if onnx_files:
    print("\n🎉 모델 파일 준비 완료!")
else:
    print("\n❌ ONNX 모델 파일이 없습니다.")
