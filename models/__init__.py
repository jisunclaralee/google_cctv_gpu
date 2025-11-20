# Models package
# 기존 시뮬레이션 모델들은 비활성화
# from .retinaface import RetinaFaceDetector
# from .arcface import ArcFaceRecognizer, EmbeddingDatabase

# 실제 AI 모델만 사용
from .real_face_detector import RealFaceDetector, get_face_detector

__all__ = ['RealFaceDetector', 'get_face_detector']
