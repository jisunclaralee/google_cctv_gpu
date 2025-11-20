"""
RetinaFace 얼굴 검출기 모듈
InsightFace 기반 PyTorch/ONNX 검출기
"""

import cv2
import numpy as np
import onnxruntime
import torch
import logging
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

class RetinaFaceDetector:
    """RetinaFace 기반 얼굴 검출기"""
    
    def __init__(self, model_path: str, gpu_enabled: bool = True, confidence_threshold: float = 0.5):
        """
        RetinaFace 검출기 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            gpu_enabled: GPU 사용 여부
            confidence_threshold: 검출 신뢰도 임계값
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.session = None
        self.input_size = (640, 640)  # RetinaFace 기본 입력 크기
        
        self._load_model()
    
    def _load_model(self):
        """ONNX 모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            # ONNX Runtime 설정
            providers = []
            if self.gpu_enabled:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self.session = onnxruntime.InferenceSession(
                self.model_path, 
                providers=providers
            )
            
            # 입력/출력 이름 확인
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"✅ RetinaFace 모델 로드 완료: {self.model_path}")
            logger.info(f"🖥️  실행 환경: {'GPU' if self.gpu_enabled else 'CPU'}")
            
        except Exception as e:
            logger.error(f"❌ RetinaFace 모델 로드 실패: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            전처리된 이미지, x 스케일, y 스케일
        """
        height, width = image.shape[:2]
        
        # 비율 유지하며 리사이즈
        scale_x = self.input_size[0] / width
        scale_y = self.input_size[1] / height
        scale = min(scale_x, scale_y)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 이미지 리사이즈
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # 패딩 추가 (중앙 정렬)
        pad_x = (self.input_size[0] - new_width) // 2
        pad_y = (self.input_size[1] - new_height) // 2
        
        padded_image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_image
        
        # 정규화 및 차원 변환
        blob = cv2.dnn.blobFromImage(
            padded_image, 
            scalefactor=1.0/127.5, 
            size=self.input_size,
            mean=(127.5, 127.5, 127.5), 
            swapRB=True
        )
        
        return blob, scale, scale
    
    def postprocess_outputs(self, outputs: List[np.ndarray], scale_x: float, scale_y: float, 
                          original_shape: Tuple[int, int]) -> List[dict]:
        """
        모델 출력 후처리
        
        Args:
            outputs: 모델 출력
            scale_x, scale_y: 스케일링 팩터
            original_shape: 원본 이미지 크기 (height, width)
            
        Returns:
            검출 결과 리스트
        """
        detections = []
        
        # RetinaFace 출력 파싱 (구현에 따라 다를 수 있음)
        if len(outputs) >= 3:
            boxes = outputs[0]  # 바운딩 박스
            scores = outputs[1]  # 신뢰도
            landmarks = outputs[2] if len(outputs) > 2 else None  # 랜드마크
            
            # 각 검출 결과 처리
            for i in range(boxes.shape[0]):
                confidence = float(scores[i])
                
                if confidence >= self.confidence_threshold:
                    # 바운딩 박스 좌표 복원
                    x1, y1, x2, y2 = boxes[i]
                    
                    # 원본 이미지 좌표로 변환
                    x1 = int(x1 / scale_x)
                    y1 = int(y1 / scale_y)
                    x2 = int(x2 / scale_x)
                    y2 = int(y2 / scale_y)
                    
                    # 경계값 클리핑
                    x1 = max(0, min(x1, original_shape[1]))
                    y1 = max(0, min(y1, original_shape[0]))
                    x2 = max(0, min(x2, original_shape[1]))
                    y2 = max(0, min(y2, original_shape[0]))
                    
                    detection = {
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                        'confidence': confidence,
                        'landmarks': None
                    }
                    
                    # 랜드마크가 있는 경우 추가
                    if landmarks is not None and i < landmarks.shape[0]:
                        lm = landmarks[i].reshape(-1, 2)
                        # 좌표 변환
                        lm[:, 0] = lm[:, 0] / scale_x
                        lm[:, 1] = lm[:, 1] / scale_y
                        detection['landmarks'] = lm.tolist()
                    
                    detections.append(detection)
        
        return detections
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        얼굴 검출 실행
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            검출 결과 리스트
        """
        if self.session is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        try:
            # 전처리
            blob, scale_x, scale_y = self.preprocess_image(image)
            
            # 추론 실행
            outputs = self.session.run(
                self.output_names, 
                {self.input_name: blob}
            )
            
            # 후처리
            detections = self.postprocess_outputs(
                outputs, scale_x, scale_y, image.shape[:2]
            )
            
            logger.debug(f"🔍 {len(detections)}개의 얼굴 검출됨")
            return detections
            
        except Exception as e:
            logger.error(f"얼굴 검출 중 오류: {str(e)}")
            return []
    
    def extract_face_roi(self, image: np.ndarray, bbox: List[int], 
                        padding: float = 0.2) -> Optional[np.ndarray]:
        """
        바운딩 박스에서 얼굴 영역 추출
        
        Args:
            image: 원본 이미지
            bbox: 바운딩 박스 [x, y, width, height]
            padding: 패딩 비율
            
        Returns:
            추출된 얼굴 이미지
        """
        try:
            x, y, w, h = bbox
            
            # 패딩 추가
            pad_x = int(w * padding / 2)
            pad_y = int(h * padding / 2)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            
            # 얼굴 영역 추출
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
                
            return face_roi
            
        except Exception as e:
            logger.error(f"얼굴 영역 추출 중 오류: {str(e)}")
            return None