"""
얼굴 인식 파이프라인
이미지 → 얼굴 검출 → 얼굴 crop/정규화 → ArcFace 임베딩 → 결과 반환
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple
import os

from models.retinaface import RetinaFaceDetector
from models.arcface import ArcFaceRecognizer, EmbeddingDatabase

logger = logging.getLogger(__name__)

class FacePipeline:
    """통합 얼굴 인식 파이프라인"""
    
    def __init__(self, gpu_enabled: bool = True, model_path: str = None, embeddings_path: str = None):
        """
        파이프라인 초기화
        
        Args:
            gpu_enabled: GPU 사용 여부
            model_path: 모델 파일 경로
            embeddings_path: 임베딩 데이터베이스 경로
        """
        self.gpu_enabled = gpu_enabled
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        
        # 기본 경로 설정
        if not self.model_path:
            self.model_path = "models/weights"
        if not self.embeddings_path:
            self.embeddings_path = "data/embeddings"
        
        # 모델 초기화
        self.detector = None
        self.recognizer = None
        self.embeddings_db = None
        
        # 설정
        self.detection_confidence = 0.5
        self.similarity_threshold = 0.6
        
        self._initialize_models()
    
    def _initialize_models(self):
        """모델 초기화"""
        try:
            # RetinaFace 검출기 초기화
            detector_path = os.path.join(self.model_path, "det_10g.onnx")
            if os.path.exists(detector_path):
                self.detector = RetinaFaceDetector(
                    model_path=detector_path,
                    gpu_enabled=self.gpu_enabled,
                    confidence_threshold=self.detection_confidence
                )
                logger.info("✅ RetinaFace 검출기 초기화 완료")
            else:
                logger.warning(f"⚠️  검출기 모델 파일이 없습니다: {detector_path}")
            
            # ArcFace 인식기 초기화
            recognizer_path = os.path.join(self.model_path, "w600k_r50.onnx")
            if os.path.exists(recognizer_path):
                self.recognizer = ArcFaceRecognizer(
                    model_path=recognizer_path,
                    gpu_enabled=self.gpu_enabled
                )
                logger.info("✅ ArcFace 인식기 초기화 완료")
            else:
                logger.warning(f"⚠️  인식기 모델 파일이 없습니다: {recognizer_path}")
            
            # 임베딩 데이터베이스 초기화
            embeddings_file = os.path.join(self.embeddings_path, "all_embeddings.json")
            suspects_file = "data/suspects/metadata/suspect_profiles.json"
            
            if os.path.exists(embeddings_file) and os.path.exists(suspects_file):
                self.embeddings_db = EmbeddingDatabase(
                    embeddings_path=embeddings_file,
                    suspects_metadata_path=suspects_file
                )
                logger.info("✅ 임베딩 데이터베이스 초기화 완료")
            else:
                logger.warning(f"⚠️  데이터베이스 파일이 없습니다")
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {str(e)}")
    
    def process_image(self, image: np.ndarray, target_suspect_id: str = None) -> List[Dict]:
        """
        단일 이미지에서 얼굴 인식 수행
        
        Args:
            image: 입력 이미지 (BGR)
            target_suspect_id: 특정 용의자 ID (선택사항)
            
        Returns:
            인식 결과 리스트
        """
        start_time = time.time()
        results = []
        
        try:
            if self.detector is None:
                raise RuntimeError("검출기가 초기화되지 않았습니다")
            
            # 1. 얼굴 검출
            detections = self.detector.detect_faces(image)
            logger.debug(f"🔍 {len(detections)}개의 얼굴 검출됨")
            
            if not detections:
                return []
            
            # 2. 각 검출된 얼굴에 대해 인식 수행
            for i, detection in enumerate(detections):
                try:
                    # 얼굴 영역 추출
                    face_roi = self.detector.extract_face_roi(image, detection['bbox'])
                    if face_roi is None:
                        continue
                    
                    # 얼굴 인식 수행
                    recognition_result = self._recognize_face(
                        face_roi, 
                        target_suspect_id=target_suspect_id
                    )
                    
                    # 결과 구성
                    result = {
                        'face_id': i,
                        'face_bbox': detection['bbox'],
                        'detection_confidence': detection['confidence'],
                        'landmarks': detection.get('landmarks'),
                        'suspect_match': recognition_result,
                        'processing_time': time.time() - start_time
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"얼굴 {i} 처리 중 오류: {str(e)}")
                    continue
            
            logger.info(f"🎯 이미지 처리 완료: {len(results)}개 얼굴, {time.time() - start_time:.2f}초")
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
        
        return results
    
    def _recognize_face(self, face_image: np.ndarray, target_suspect_id: str = None) -> Optional[Dict]:
        """
        얼굴 인식 수행
        
        Args:
            face_image: 얼굴 이미지
            target_suspect_id: 특정 용의자 ID
            
        Returns:
            인식 결과
        """
        try:
            if self.recognizer is None or self.embeddings_db is None:
                return None
            
            # 임베딩 추출
            face_embedding = self.recognizer.extract_embedding(face_image)
            if face_embedding is None:
                return None
            
            # 데이터베이스와 비교
            if target_suspect_id:
                # 특정 용의자와 비교
                target_embedding = self.embeddings_db.get_embedding(target_suspect_id)
                if target_embedding is not None:
                    similarity = self.recognizer.compare_embeddings(
                        face_embedding, target_embedding
                    )
                    
                    if similarity >= self.similarity_threshold:
                        suspect_info = self.embeddings_db.get_suspect_info(target_suspect_id)
                        
                        return {
                            'suspect_id': target_suspect_id,
                            'name': suspect_info.get('name', 'Unknown') if suspect_info else 'Unknown',
                            'similarity': similarity,
                            'confidence': similarity * 100,
                            'is_criminal': suspect_info.get('is_criminal', False) if suspect_info else False,
                            'risk_level': suspect_info.get('risk_level', 'unknown') if suspect_info else 'unknown',
                            'criminal_record': suspect_info.get('criminal_record', []) if suspect_info else [],
                            'category': suspect_info.get('role', 'unknown') if suspect_info else 'unknown'
                        }
            else:
                # 전체 데이터베이스와 비교
                all_embeddings = self.embeddings_db.get_all_embeddings()
                best_match = self.recognizer.find_most_similar(
                    face_embedding, all_embeddings, self.similarity_threshold
                )
                
                if best_match:
                    suspect_info = self.embeddings_db.get_suspect_info(best_match['suspect_id'])
                    
                    return {
                        'suspect_id': best_match['suspect_id'],
                        'name': suspect_info.get('name', 'Unknown') if suspect_info else 'Unknown',
                        'similarity': best_match['similarity'],
                        'confidence': best_match['confidence'],
                        'is_criminal': suspect_info.get('is_criminal', False) if suspect_info else False,
                        'risk_level': suspect_info.get('risk_level', 'unknown') if suspect_info else 'unknown',
                        'criminal_record': suspect_info.get('criminal_record', []) if suspect_info else [],
                        'category': suspect_info.get('role', 'unknown') if suspect_info else 'unknown'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"얼굴 인식 중 오류: {str(e)}")
            return None
    
    def process_video(self, video_path: str, target_suspect_id: str = None, 
                     frame_interval: int = 30) -> List[Dict]:
        """
        비디오 파일에서 얼굴 인식 수행
        
        Args:
            video_path: 비디오 파일 경로
            target_suspect_id: 특정 용의자 ID
            frame_interval: 프레임 간격 (몇 프레임마다 처리할지)
            
        Returns:
            프레임별 인식 결과
        """
        video_results = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"📹 비디오 처리 시작: {total_frames} 프레임, {fps} FPS")
            
            frame_number = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 지정된 간격마다 프레임 처리
                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps
                    
                    # 얼굴 인식 수행
                    frame_results = self.process_image(frame, target_suspect_id)
                    
                    if frame_results:
                        video_results.append({
                            'frame_number': frame_number,
                            'timestamp': timestamp,
                            'detections': frame_results
                        })
                    
                    processed_frames += 1
                    
                    if processed_frames % 10 == 0:
                        logger.info(f"🎬 처리 중: {processed_frames} / {total_frames // frame_interval} 프레임")
                
                frame_number += 1
            
            cap.release()
            
            logger.info(f"✅ 비디오 처리 완료: {len(video_results)}개 프레임에서 검출")
            
        except Exception as e:
            logger.error(f"비디오 처리 중 오류: {str(e)}")
        
        return video_results
    
    def update_threshold(self, detection_threshold: float = None, 
                        similarity_threshold: float = None):
        """임계값 업데이트"""
        if detection_threshold is not None:
            self.detection_confidence = detection_threshold
            if self.detector:
                self.detector.confidence_threshold = detection_threshold
        
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
    
    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            'detector_loaded': self.detector is not None,
            'recognizer_loaded': self.recognizer is not None,
            'database_loaded': self.embeddings_db is not None,
            'gpu_enabled': self.gpu_enabled,
            'detection_confidence': self.detection_confidence,
            'similarity_threshold': self.similarity_threshold,
            'suspects_count': len(self.embeddings_db) if self.embeddings_db else 0
        }