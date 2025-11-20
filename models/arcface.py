"""
ArcFace 얼굴 인식 모듈
InsightFace 기반 얼굴 임베딩 생성
"""

import cv2
import numpy as np
import onnxruntime
import torch
import logging
from typing import List, Optional, Dict
import os
import json

logger = logging.getLogger(__name__)

class ArcFaceRecognizer:
    """ArcFace 기반 얼굴 인식기"""
    
    def __init__(self, model_path: str, gpu_enabled: bool = True):
        """
        ArcFace 인식기 초기화
        
        Args:
            model_path: ONNX 모델 파일 경로
            gpu_enabled: GPU 사용 여부
        """
        self.model_path = model_path
        self.gpu_enabled = gpu_enabled and torch.cuda.is_available()
        self.session = None
        self.input_size = (112, 112)  # ArcFace 표준 입력 크기
        self.embedding_dim = 512  # ArcFace 임베딩 차원
        
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
            
            # 입력/출력 정보 확인
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # 입력 크기 확인 (동적으로 조정)
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 3:
                self.input_size = (input_shape[2], input_shape[3])
            
            logger.info(f"✅ ArcFace 모델 로드 완료: {self.model_path}")
            logger.info(f"🖥️  실행 환경: {'GPU' if self.gpu_enabled else 'CPU'}")
            logger.info(f"📐 입력 크기: {self.input_size}")
            
        except Exception as e:
            logger.error(f"❌ ArcFace 모델 로드 실패: {str(e)}")
            raise
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지 전처리
        
        Args:
            face_image: 얼굴 이미지 (BGR)
            
        Returns:
            전처리된 이미지
        """
        try:
            # 크기 조정
            face_resized = cv2.resize(face_image, self.input_size)
            
            # 정규화 및 차원 변환
            blob = cv2.dnn.blobFromImage(
                face_resized,
                scalefactor=1.0/127.5,
                size=self.input_size,
                mean=(127.5, 127.5, 127.5),
                swapRB=True
            )
            
            return blob
            
        except Exception as e:
            logger.error(f"얼굴 이미지 전처리 중 오류: {str(e)}")
            return None
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        얼굴 이미지에서 임베딩 벡터 추출
        
        Args:
            face_image: 얼굴 이미지 (BGR)
            
        Returns:
            임베딩 벡터 (512차원)
        """
        if self.session is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        
        try:
            # 전처리
            blob = self.preprocess_face(face_image)
            if blob is None:
                return None
            
            # 추론 실행
            embedding = self.session.run(
                [self.output_name], 
                {self.input_name: blob}
            )[0]
            
            # 정규화
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 추출 중 오류: {str(e)}")
            return None
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 간의 코사인 유사도 계산
        
        Args:
            embedding1, embedding2: 비교할 임베딩 벡터
            
        Returns:
            코사인 유사도 (0~1)
        """
        try:
            # 코사인 유사도 계산
            similarity = np.dot(embedding1, embedding2)
            
            # 클리핑 (수치 안정성)
            similarity = np.clip(similarity, -1.0, 1.0)
            
            # 0~1 범위로 변환
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"임베딩 비교 중 오류: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         database_embeddings: Dict[str, np.ndarray],
                         threshold: float = 0.6) -> Optional[Dict]:
        """
        데이터베이스에서 가장 유사한 임베딩 찾기
        
        Args:
            query_embedding: 쿼리 임베딩
            database_embeddings: 데이터베이스 임베딩들
            threshold: 유사도 임계값
            
        Returns:
            가장 유사한 결과 정보
        """
        try:
            best_match = None
            best_similarity = 0.0
            
            for suspect_id, db_embedding in database_embeddings.items():
                similarity = self.compare_embeddings(query_embedding, db_embedding)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = {
                        'suspect_id': suspect_id,
                        'similarity': similarity,
                        'confidence': similarity * 100
                    }
            
            return best_match
            
        except Exception as e:
            logger.error(f"유사도 검색 중 오류: {str(e)}")
            return None

class EmbeddingDatabase:
    """얼굴 임베딩 데이터베이스 관리"""
    
    def __init__(self, embeddings_path: str, suspects_metadata_path: str):
        """
        임베딩 데이터베이스 초기화
        
        Args:
            embeddings_path: 임베딩 파일 경로
            suspects_metadata_path: 용의자 메타데이터 경로
        """
        self.embeddings_path = embeddings_path
        self.suspects_metadata_path = suspects_metadata_path
        self.embeddings = {}
        self.suspects_info = {}
        
        self._load_database()
    
    def _load_database(self):
        """데이터베이스 로드"""
        try:
            # 임베딩 데이터 로드
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                
                # numpy 배열로 변환
                for suspect_id, embedding_list in embeddings_data.items():
                    if suspect_id != 'metadata':
                        self.embeddings[suspect_id] = np.array(embedding_list)
                
                logger.info(f"✅ 임베딩 데이터 로드 완료: {len(self.embeddings)}명")
            
            # 용의자 메타데이터 로드
            if os.path.exists(self.suspects_metadata_path):
                with open(self.suspects_metadata_path, 'r', encoding='utf-8') as f:
                    suspects_data = json.load(f)
                
                # 용의자 정보 인덱싱
                for suspect in suspects_data.get('suspects', []):
                    self.suspects_info[suspect['id']] = suspect
                
                logger.info(f"✅ 용의자 메타데이터 로드 완료: {len(self.suspects_info)}명")
            
        except Exception as e:
            logger.error(f"데이터베이스 로드 중 오류: {str(e)}")
    
    def get_embedding(self, suspect_id: str) -> Optional[np.ndarray]:
        """용의자 임베딩 조회"""
        return self.embeddings.get(suspect_id)
    
    def get_suspect_info(self, suspect_id: str) -> Optional[Dict]:
        """용의자 정보 조회"""
        return self.suspects_info.get(suspect_id)
    
    def add_embedding(self, suspect_id: str, embedding: np.ndarray):
        """임베딩 추가"""
        self.embeddings[suspect_id] = embedding
    
    def save_database(self):
        """데이터베이스 저장"""
        try:
            # 임베딩을 리스트로 변환하여 저장
            embeddings_to_save = {}
            for suspect_id, embedding in self.embeddings.items():
                embeddings_to_save[suspect_id] = embedding.tolist()
            
            # 메타데이터 추가
            embeddings_to_save['metadata'] = {
                'saved_date': str(np.datetime64('now')),
                'total_embeddings': len(self.embeddings)
            }
            
            with open(self.embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 임베딩 데이터베이스 저장 완료: {self.embeddings_path}")
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 중 오류: {str(e)}")
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """모든 임베딩 반환"""
        return self.embeddings.copy()
    
    def __len__(self):
        """데이터베이스 크기 반환"""
        return len(self.embeddings)