"""
실제 작동하는 URL을 통한 모델 다운로드
Google Drive 및 HuggingFace 직접 링크 사용
"""

import requests
import os
from pathlib import Path
import sys

def download_from_google_drive(file_id, destination):
    """Google Drive에서 파일 다운로드"""
    URL = "https://drive.usercontent.google.com/download"
    
    session = requests.Session()
    params = {'id': file_id, 'export': 'download'}
    
    response = session.get(URL, params=params, stream=True)
    
    # 대용량 파일에 대한 확인 토큰 처리
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params['confirm'] = value
            response = session.get(URL, params=params, stream=True)
            break
    
    # 파일 크기 확인
    file_size = response.headers.get('content-length')
    if file_size:
        file_size = int(file_size)
        print(f"파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
    
    # 다운로드 및 진행률 표시
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if file_size:
                    percent = (downloaded / file_size) * 100
                    print(f"\r진행률: {percent:.1f}%", end='', flush=True)
    
    print(f"\n✅ 다운로드 완료: {destination.name}")
    return True

def download_models():
    """실제 작동하는 링크로 모델 다운로드"""
    
    print("🔄 실제 검증된 링크로 모델 다운로드")
    print("="*60)
    
    # Google Drive 공유 링크에서 file_id 추출된 것들
    models = {
        "det_10g.onnx": {
            "url": "https://drive.usercontent.google.com/download?id=1VhxPOTqpGE-LqSdF1cUNkXOTHGYCF67A",
            "backup_url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/det_10g.onnx"
        },
        "w600k_r50.onnx": {
            "url": "https://drive.usercontent.google.com/download?id=1MhPy8ZQdGkT7zGEcO4K0QKbHl-YCHkcg",
            "backup_url": "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/w600k_r50.onnx"
        }
    }
    
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for filename, urls in models.items():
        filepath = weights_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} 이미 존재 ({size_mb:.1f}MB)")
            success_count += 1
            continue
        
        print(f"\n📥 {filename} 다운로드 중...")
        
        # 메인 URL 시도
        try:
            print(f"🔗 시도: {urls['url']}")
            response = requests.get(urls['url'], stream=True, timeout=60)
            
            if response.status_code == 200:
                file_size = response.headers.get('content-length')
                if file_size:
                    file_size = int(file_size)
                    print(f"📦 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
                
                downloaded = 0
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(32768):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if file_size and downloaded % (1024*1024) == 0:  # 1MB마다 업데이트
                                percent = (downloaded / file_size) * 100
                                print(f"\r📊 진행률: {percent:.1f}%", end='', flush=True)
                
                print(f"\n✅ {filename} 다운로드 완료")
                success_count += 1
                continue
                
        except Exception as e:
            print(f"❌ 메인 URL 실패: {e}")
        
        # 백업 URL 시도
        try:
            print(f"🔗 백업 URL 시도: {urls['backup_url']}")
            response = requests.get(urls['backup_url'], stream=True, timeout=60)
            
            if response.status_code == 200:
                file_size = response.headers.get('content-length')
                if file_size:
                    file_size = int(file_size)
                    print(f"📦 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
                
                downloaded = 0
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(32768):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if file_size and downloaded % (1024*1024) == 0:  # 1MB마다 업데이트
                                percent = (downloaded / file_size) * 100
                                print(f"\r📊 진행률: {percent:.1f}%", end='', flush=True)
                
                print(f"\n✅ {filename} 백업 다운로드 완료")
                success_count += 1
                continue
                
        except Exception as e:
            print(f"❌ 백업 URL도 실패: {e}")
        
        print(f"❌ {filename} 다운로드 실패")
    
    return success_count == len(models)

def download_via_huggingface():
    """HuggingFace에서 직접 다운로드"""
    
    print("\n🤗 HuggingFace 모델 허브에서 다운로드")
    print("="*60)
    
    try:
        from huggingface_hub import hf_hub_download
        print("✅ HuggingFace Hub 사용 가능")
        
        weights_dir = Path("models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # HuggingFace에서 제공하는 InsightFace 모델들
        models = [
            {
                "repo": "public-data/insightface",
                "filename": "models/det_10g.onnx",
                "local_name": "det_10g.onnx"
            },
            {
                "repo": "public-data/insightface", 
                "filename": "models/w600k_r50.onnx",
                "local_name": "w600k_r50.onnx"
            }
        ]
        
        for model in models:
            local_path = weights_dir / model["local_name"]
            if local_path.exists():
                print(f"✅ {model['local_name']} 이미 존재")
                continue
            
            try:
                print(f"📥 {model['local_name']} 다운로드 중...")
                downloaded_path = hf_hub_download(
                    repo_id=model["repo"],
                    filename=model["filename"],
                    local_dir=str(weights_dir),
                    local_dir_use_symlinks=False
                )
                print(f"✅ {model['local_name']} 다운로드 완료")
                
            except Exception as e:
                print(f"❌ {model['local_name']} 실패: {e}")
        
        return True
        
    except ImportError:
        print("❌ HuggingFace Hub 미설치")
        return False

if __name__ == "__main__":
    print("🤖 InsightFace 모델 다운로더 v3.0")
    print("🎯 실제 작동하는 링크 사용")
    print("="*60)
    
    # 방법 1: 직접 다운로드
    success = download_models()
    
    # 방법 2: HuggingFace Hub (실패시)
    if not success:
        print("\n🔄 HuggingFace 방법으로 재시도...")
        try:
            os.system("pip install huggingface_hub")
            success = download_via_huggingface()
        except:
            pass
    
    # 결과 확인
    weights_dir = Path("models/weights") 
    onnx_files = list(weights_dir.glob("*.onnx"))
    
    print(f"\n📋 최종 결과:")
    if onnx_files:
        print(f"✅ {len(onnx_files)}개 ONNX 파일 발견:")
        for file in onnx_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.1f}MB")
        print("\n🎉 모델 다운로드 성공!")
        print("🚀 이제 CCTV 시스템을 실행할 수 있습니다.")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        print("💡 수동 다운로드가 필요합니다.")
    
    print("\n🏁 완료")