import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
PROFILES_PATH = PROJECT_ROOT / "data" / "suspects" / "metadata" / "suspect_profiles.json"


class RealFaceDetector:
    """실제 InsightFace AI 모델을 사용한 얼굴 인식 시스템"""
    
    def __init__(self):
        """AI 모델 초기화"""
        print("Initializing Real AI Face Detection System...")
        
        # InsightFace FaceAnalysis 초기화
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # 용의자 임베딩 데이터베이스 로드
        self.suspect_embeddings = self._load_suspect_embeddings()
        self.suspect_profiles = self._load_suspect_profiles()
        self.suspect_profile_lookup = self._build_profile_lookup()
        self._merge_profile_metadata()
        
        print("Real AI Face Detection System initialized successfully!")
        print(f"Loaded {len(self.suspect_embeddings)} suspect profiles")
    
    def _load_suspect_embeddings(self) -> Dict:
        """용의자 임베딩 데이터베이스 로드 (모든 임베딩 활용)"""
        embeddings_db: Dict[str, Dict] = {}

        def _normalize_vector(vector: List[float]) -> np.ndarray:
            arr = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0:
                return arr
            return arr / norm

        def _upsert_person(person_id: Optional[str], name: str, info: Dict, vectors: List[List[float]]):
            if not person_id or not vectors:
                return
            normalized_vectors = [_normalize_vector(vec) for vec in vectors]
            person_type = 'criminal' if (info or {}).get('is_criminal') else (info or {}).get('role', 'unknown')
            if person_type == 'unknown' and 'criminal' in (person_id or '').lower():
                person_type = 'criminal'
            elif person_type == 'unknown' and 'normal' in (person_id or '').lower():
                person_type = 'normal'
            if person_id not in embeddings_db:
                embeddings_db[person_id] = {
                    'name': name or person_id,
                    'info': info or {},
                    'embeddings': normalized_vectors,
                    'type': person_type
                }
            else:
                embeddings_db[person_id]['embeddings'].extend(normalized_vectors)
                if name:
                    embeddings_db[person_id]['name'] = name
                if info:
                    embeddings_db[person_id]['info'].update(info)
                if person_type != 'unknown':
                    embeddings_db[person_id]['type'] = person_type

        try:
            all_embeddings_path = EMBEDDINGS_DIR / 'all_embeddings.json'
            if all_embeddings_path.exists():
                with open(all_embeddings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for person in data.get('persons', []):
                    person_id = person.get('person_id')
                    embeddings_map = person.get('embeddings', {})
                    _upsert_person(
                        person_id,
                        person.get('name', person_id),
                        person.get('info', {}),
                        list(embeddings_map.values())
                    )

            # 추가 개별 임베딩 파일 (criminal, normal 등) 병합
            for emb_file in EMBEDDINGS_DIR.glob('*_embeddings.json'):
                if emb_file.name == 'all_embeddings.json':
                    continue
                with open(emb_file, 'r', encoding='utf-8') as f:
                    personal_data = json.load(f)
                person_id = personal_data.get('person_id') or emb_file.stem.replace('_embeddings', '')
                embeddings_map = personal_data.get('embeddings', {})
                _upsert_person(
                    person_id,
                    personal_data.get('name', person_id),
                    personal_data.get('info', {}),
                    list(embeddings_map.values())
                )

            print(f"Loaded {len(embeddings_db)} suspect embeddings from {EMBEDDINGS_DIR}")
            return embeddings_db

        except FileNotFoundError:
            print(f"Warning: Embeddings directory not found at {EMBEDDINGS_DIR}")
            return {}
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return {}
    
    def _load_suspect_profiles(self) -> Dict:
        """용의자 프로필 정보 로드"""
        try:
            with open(PROFILES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # suspects 배열을 id로 인덱싱
                profiles = {}
                for suspect in data.get('suspects', []):
                    profiles[suspect['id']] = suspect
                return profiles
        except FileNotFoundError:
            print(f"Warning: Profiles file not found at {PROFILES_PATH}")
            return {}
        except Exception as e:
            print(f"Error loading profiles: {e}")
            return {}
    
    def _build_profile_lookup(self) -> Dict[str, str]:
        """embedding �� person_id�� ���� suspect id mapping"""
        lookup: Dict[str, str] = {}
        try:
            for suspect_id, profile in (self.suspect_profiles or {}).items():
                folder_name = profile.get('folder_name')
                if folder_name:
                    lookup[folder_name.lower()] = suspect_id
                name_en = profile.get('name_en')
                if name_en:
                    lookup[name_en.lower()] = suspect_id
                lookup[str(suspect_id).lower()] = suspect_id
        except Exception as e:
            print(f"Error building suspect profile lookup: {e}")
        return lookup

    def _merge_profile_metadata(self) -> None:
        """임베딩 정보에 프로필 메타데이터 병합"""
        if not self.suspect_embeddings or not self.suspect_profiles:
            return
        for suspect_id, profile in self.suspect_profiles.items():
            if suspect_id in self.suspect_embeddings:
                entry = self.suspect_embeddings[suspect_id]
                entry.setdefault('info', {}).update(profile)
                entry['type'] = 'criminal' if profile.get('is_criminal') else 'normal'
    def get_suspect_profile(self, suspect_id: str) -> Optional[Dict]:
        """특정 용의자 프로필 정보 반환"""
        return self.suspect_profiles.get(suspect_id)

    def get_all_suspect_profiles(self) -> Dict:
        """모든 용의자 프로필 정보 반환"""
        return self.suspect_profiles
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        실제 AI 모델로 얼굴 탐지 및 인식
        
        Args:
            frame: 입력 이미지 프레임 (BGR 형식 - InsightFace 요구사항)
            
        Returns:
            탐지된 얼굴 정보 리스트
        """
        try:
            print(f"🔍 InsightFace 입력: {frame.shape}, dtype: {frame.dtype}")
            
            # InsightFace로 얼굴 분석 (BGR 형식 기대)
            faces = self.app.get(frame)
            
            print(f"✅ InsightFace 결과: {len(faces)}개 얼굴 감지됨")
            
            detected_faces = []
            
            for face in faces:
                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 얼굴 임베딩
                embedding = face.embedding
                
                # 용의자 매칭
                suspect_id, confidence = self._match_suspect(embedding)
                
                # 얼굴 정보 구성
                match_conf = float(confidence) if confidence is not None else 0.0
                face_info = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(face.det_score),
                    'suspect_id': suspect_id,
                    'match_confidence': match_conf,
                    'recognition_confidence': match_conf,
                    'is_suspect': suspect_id is not None,
                    'age': int(face.age) if hasattr(face, 'age') else None,
                    'gender': 'Male' if face.gender == 1 else 'Female' if hasattr(face, 'gender') else None,
                    'person_name': None,
                    'person_type': 'unknown',
                    'is_normal': False
                }
                
                # 용의자 정보 추가 (실제 suspect_profiles.json 구조)
                if suspect_id and suspect_id in self.suspect_profiles:
                    profile = self.suspect_profiles[suspect_id]
                    face_info.update({
                        'name': profile.get('name', '알 수 없음'),
                        'risk_level': profile.get('risk_level', 'low'),
                        'is_criminal': profile.get('is_criminal', False),
                        'criminal_record': profile.get('criminal_record', []),
                        'occupation': profile.get('occupation', '알 수 없음'),
                        'suspect_category': 'criminal' if profile.get('is_criminal', False) else 'civilian',
                        'danger_level': 'HIGH' if profile.get('risk_level') == 'high' else 'LOW',
                        'person_name': profile.get('name', '알 수 없음'),
                        'person_type': 'criminal' if profile.get('is_criminal', False) else 'normal',
                        'is_normal': not profile.get('is_criminal', False)
                    })
                    print(f"🎯 용의자 매칭: {profile.get('name')} (ID: {suspect_id}, 위험도: {profile.get('risk_level')})")
                else:
                    # 매칭되지 않은 일반인
                    face_info.update({
                        'name': None,
                        'risk_level': 'low',
                        'is_criminal': False,
                        'criminal_record': [],
                        'occupation': '알 수 없음',
                        'suspect_category': 'unknown',
                        'danger_level': 'LOW',
                        'person_name': '미확인 인물',
                        'person_type': 'unknown',
                        'is_normal': False
                    })
                
                detected_faces.append(face_info)
            
            return detected_faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def _match_suspect(self, face_embedding: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        얼굴 임베딩을 용의자 데이터베이스와 매칭
        
        Args:
            face_embedding: 얼굴 임베딩 벡터
            
        Returns:
            (용의자 ID, 유사도) 또는 (None, None)
        """
        if not self.suspect_embeddings:
            return None, None
        
        best_match = None
        best_similarity = 0.0
        threshold = 0.6  # 유사도 임계값
        
        try:
            face_norm = face_embedding / np.linalg.norm(face_embedding)
            for person_id, person_data in self.suspect_embeddings.items():
                for stored_embedding in person_data.get('embeddings', []):
                    stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                    
                    similarity = np.dot(face_norm, stored_norm)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = person_id
                        print(f"🎯 매칭 후보: {person_id} ({person_data.get('name')}) - 유사도: {similarity:.3f}")
            
            if best_match:
                key = best_match.lower() if isinstance(best_match, str) else best_match
                mapped_id = self.suspect_profile_lookup.get(key, best_match)
                print(f"✅ 최종 매칭: {best_match} -> {mapped_id}, 유사도: {best_similarity:.3f}")
                return mapped_id, best_similarity
                
            return None, None
            
        except Exception as e:
            print(f"Error matching suspect: {e}")
            return None, None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        try:
            # 벡터 정규화
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            
            if a_norm == 0 or b_norm == 0:
                return 0.0
            
            # 코사인 유사도
            similarity = np.dot(a, b) / (a_norm * b_norm)
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        프레임 처리 및 결과 반환
        
        Args:
            frame: 입력 프레임
            
        Returns:
            처리 결과
        """
        # 얼굴 탐지
        faces = self.detect_faces(frame)
        
        # 통계 계산
        total_faces = len(faces)
        suspect_faces = len([f for f in faces if f['is_suspect']])
        
        # 알림 레벨 결정
        alert_level = 'high' if suspect_faces > 0 else 'normal'
        
        return {
            'faces': faces,
            'total_faces': total_faces,
            'suspect_faces': suspect_faces,
            'alert_level': alert_level,
            'timestamp': np.datetime64('now').astype(str)
        }

# 전역 인스턴스
_face_detector = None

def get_face_detector() -> RealFaceDetector:
    """얼굴 탐지기 싱글톤 인스턴스 반환"""
    global _face_detector
    if _face_detector is None:
        _face_detector = RealFaceDetector()
    return _face_detector

def detect_faces_in_frame(frame: np.ndarray) -> Dict:
    """프레임에서 얼굴 탐지 (외부 API용)"""
    detector = get_face_detector()
    return detector.process_frame(frame)
