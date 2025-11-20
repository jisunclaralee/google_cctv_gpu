"""
CCTV 용의자 식별 시스템 설정 파일
"""

import os
from pathlib import Path

class Config:
    """시스템 설정 클래스"""
    
    # 기본 디렉토리 설정
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    UPLOAD_DIR = BASE_DIR / "uploads"
    
    # 서버 설정
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5000
    DEBUG_MODE = True
    
    # GPU 설정
    USE_GPU = True  # GPU 사용 여부 (CUDA 사용 가능시에만)
    GPU_MEMORY_FRACTION = 0.7  # GPU 메모리 사용량 제한
    
    # 모델 파일 경로
    MODEL_PATH = str(MODEL_DIR / "weights")
    RETINAFACE_MODEL = "det_10g.onnx"  # RetinaFace ONNX 모델
    ARCFACE_MODEL = "w600k_r50.onnx"   # ArcFace ONNX 모델
    
    # 데이터 경로
    EMBEDDINGS_PATH = str(DATA_DIR / "embeddings")
    SUSPECTS_METADATA_PATH = str(DATA_DIR / "suspects" / "metadata" / "suspect_profiles.json")
    UPLOAD_PATH = str(UPLOAD_DIR)
    
    # 임계값 설정
    FACE_CONFIDENCE_THRESHOLD = 0.5   # 얼굴 검출 신뢰도 임계값
    SIMILARITY_THRESHOLD = 0.6        # 얼굴 유사도 임계값
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE = (1920, 1080)     # 최대 이미지 크기
    FACE_PADDING = 0.2               # 얼굴 영역 패딩 비율
    
    # 비디오 처리 설정
    VIDEO_FRAME_INTERVAL = 30        # 처리할 프레임 간격 (기본: 30프레임마다)
    MAX_VIDEO_DURATION = 3600        # 최대 비디오 길이 (초)
    
    # 로그 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API 설정
    CORS_ORIGINS = ["*"]  # CORS 허용 도메인
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 최대 업로드 파일 크기 (100MB)
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        directories = [
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.UPLOAD_DIR,
            cls.MODEL_DIR / "weights",
            cls.DATA_DIR / "embeddings",
            cls.DATA_DIR / "suspects" / "images",
            cls.DATA_DIR / "suspects" / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 디렉토리 생성: {directory}")
    
    @classmethod
    def get_model_paths(cls):
        """모델 파일 경로 반환"""
        return {
            'retinaface': cls.MODEL_PATH / cls.RETINAFACE_MODEL,
            'arcface': cls.MODEL_PATH / cls.ARCFACE_MODEL
        }
    
    @classmethod
    def check_cuda_availability(cls):
        """CUDA 사용 가능성 확인"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                return {
                    'available': True,
                    'count': gpu_count,
                    'name': gpu_name,
                    'memory': torch.cuda.get_device_properties(0).total_memory if gpu_count > 0 else 0
                }
            else:
                return {'available': False, 'count': 0}
        except ImportError:
            return {'available': False, 'count': 0, 'error': 'PyTorch not installed'}
    
    @classmethod
    def validate_environment(cls):
        """환경 검증"""
        issues = []
        
        # 필수 디렉토리 확인
        if not cls.DATA_DIR.exists():
            issues.append(f"데이터 디렉토리가 없습니다: {cls.DATA_DIR}")
        
        # 모델 파일 확인
        model_paths = cls.get_model_paths()
        for model_name, model_path in model_paths.items():
            if not Path(model_path).exists():
                issues.append(f"{model_name} 모델 파일이 없습니다: {model_path}")
        
        # 용의자 메타데이터 확인
        if not Path(cls.SUSPECTS_METADATA_PATH).exists():
            issues.append(f"용의자 메타데이터 파일이 없습니다: {cls.SUSPECTS_METADATA_PATH}")
        
        return issues

# 환경 변수 기반 설정 오버라이드
class EnvConfig(Config):
    """환경 변수 기반 설정 (운영 환경용)"""
    
    # 환경 변수에서 설정 읽기
    USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
    
    FACE_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_CONFIDENCE_THRESHOLD', '0.5'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))
    
    # 사용자 정의 모델 경로 (환경 변수에서)
    if os.getenv('MODEL_PATH'):
        MODEL_PATH = os.getenv('MODEL_PATH')
    
    if os.getenv('DATA_PATH'):
        DATA_DIR = Path(os.getenv('DATA_PATH'))

# 개발 환경용 설정
class DevConfig(Config):
    """개발 환경 설정"""
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    USE_GPU = False  # 개발 시 CPU 사용

# 운영 환경용 설정
class ProdConfig(Config):
    """운영 환경 설정"""
    DEBUG_MODE = False
    LOG_LEVEL = "INFO"
    USE_GPU = True  # 운영 시 GPU 사용

def get_config():
    """환경에 따른 설정 반환"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProdConfig()
    elif env == 'development':
        return DevConfig()
    else:
        return EnvConfig()

if __name__ == "__main__":
    # 설정 테스트 및 환경 검증
    config = get_config()
    
    print("🔧 CCTV 용의자 식별 시스템 설정")
    print("=" * 50)
    print(f"환경: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"GPU 사용: {config.USE_GPU}")
    print(f"서버 포트: {config.SERVER_PORT}")
    print(f"모델 경로: {config.MODEL_PATH}")
    print(f"데이터 경로: {config.DATA_DIR}")
    
    # 디렉토리 생성
    config.create_directories()
    
    # 환경 검증
    issues = config.validate_environment()
    if issues:
        print("\n⚠️  환경 문제:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 환경 검증 완료")
    
    # CUDA 정보
    cuda_info = config.check_cuda_availability()
    if cuda_info['available']:
        print(f"\n🚀 GPU 정보:")
        print(f"  GPU 개수: {cuda_info['count']}")
        print(f"  GPU 이름: {cuda_info.get('name', 'Unknown')}")
        if 'memory' in cuda_info:
            memory_gb = cuda_info['memory'] / (1024**3)
            print(f"  GPU 메모리: {memory_gb:.1f} GB")
    else:
        print("\n🔧 GPU를 사용할 수 없습니다. CPU 모드로 동작합니다.")
        if 'error' in cuda_info:
            print(f"  오류: {cuda_info['error']}")