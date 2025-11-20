"""
CCTV 용의자 식별 시스템 - Flask API 서버
Real InsightFace AI Models (RetinaFace + ArcFace)
시뮬레이션 제거 - 실제 AI 모델만 사용
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import base64
import io
import sys

# 프로젝트 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 실제 AI 모델 import (시뮬레이션 제거)
from models.real_face_detector import detect_faces_in_frame, get_face_detector

# Flask 애플리케이션 초기화
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
face_detector = None
suspect_embeddings = {}

def load_suspect_embeddings():
    """임베딩 데이터 로딩"""
    global suspect_embeddings
    try:
        logger.info("용의자 임베딩 데이터 로딩 중...")
        if face_detector and face_detector.suspect_embeddings:
            suspect_embeddings = {
                suspect_id: {
                    'name': data.get('name', suspect_id),
                    'info': data.get('info', {}),
                    'embeddings': [emb.tolist() for emb in data.get('embeddings', [])],
                    'type': data.get('type', 'unknown')
                }
                for suspect_id, data in face_detector.suspect_embeddings.items()
            }
            logger.info(f"📊 총 {len(suspect_embeddings)}명의 임베딩 데이터 동기화 완료")
            return True
        
    except Exception as e:
        logger.error(f"❌ 임베딩 데이터 로딩 실패: {str(e)}")
        return False

def calculate_cosine_similarity(embedding1, embedding2):
    """코사인 유사도 계산"""
    try:
        # numpy 배열로 변환
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # 정규화
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 코사인 유사도 계산
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        logger.error(f"유사도 계산 오류: {str(e)}")
        return 0.0

def identify_person(face_embedding, threshold=0.6):
    """얼굴 임베딩으로 사람 식별"""
    if not suspect_embeddings:
        return None, 0.0, 'unknown'
    
    best_match = None
    best_similarity = 0.0
    person_type = 'unknown'
    
    try:
        for person_id, person_data in suspect_embeddings.items():
            for stored_embedding in person_data['embeddings']:
                similarity = calculate_cosine_similarity(face_embedding, stored_embedding)
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = person_data['name']
                    person_type = person_data['type']
        
        logger.info(f"식별 결과: {best_match}, 유사도: {best_similarity:.3f}, 타입: {person_type}")
        return best_match, best_similarity, person_type
        
    except Exception as e:
        logger.error(f"사람 식별 오류: {str(e)}")
        return None, 0.0, 'unknown'

def initialize_face_detector():
    """실제 AI 얼굴 인식 시스템 초기화"""
    global face_detector
    try:
        logger.info("Real AI Face Detection System 초기화 중...")
        face_detector = get_face_detector()
        logger.info("✅ Real AI Face Detection System 초기화 완료")
        
        # 임베딩 데이터 로딩
        if load_suspect_embeddings():
            logger.info("✅ 용의자 임베딩 데이터 로딩 완료")
        else:
            logger.warning("⚠️ 임베딩 데이터 로딩 실패 - 일반 얼굴 감지만 가능")
        
        return True
    except Exception as e:
        logger.error(f"❌ AI 시스템 초기화 실패: {str(e)}")
        return False

@app.route('/')
def index():
    """홈페이지"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """시스템 상태 확인"""
    try:
        # GPU 상태 확인 (옵션)
        gpu_available = False
        gpu_count = 0
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
        except ImportError:
            # PyTorch가 없어도 InsightFace는 CPU로 동작 가능
            gpu_available = False
            gpu_count = 0
        
        # AI 모델 상태
        models_ready = face_detector is not None
        
        # 용의자 데이터베이스 상태
        suspects_count = 0
        embeddings_loaded = False
        if face_detector:
            suspects_count = len(face_detector.suspect_embeddings)
            embeddings_loaded = suspects_count > 0
        
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "gpu": {
                "available": gpu_available,
                "count": gpu_count
            },
            "models": {
                "real_ai_detector": models_ready,
                "insightface_loaded": models_ready,
                "pipeline_initialized": models_ready
            },
            "database": {
                "embeddings_loaded": embeddings_loaded,
                "suspects_count": suspects_count
            },
            "message": "Real AI Face Detection System Active" if models_ready else "Initializing AI Models..."
        }
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"상태 확인 중 오류: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """웹캠 프레임에서 얼굴 인식 및 범죄자 식별"""
    try:
        if not face_detector:
            return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500
        
        # JSON 데이터 파싱
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "이미지 데이터가 없습니다."}), 400
        
        # Base64 이미지 디코딩
        try:
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            
            # PIL Image로 로드 (RGB)
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # PIL(RGB) → numpy(RGB) → BGR 변환 (InsightFace 요구사항)
            rgb_array = np.array(pil_image)
            bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"이미지 변환 완료: {bgr_image.shape}, BGR 형식으로 InsightFace에 전달")
            cv_image = bgr_image
            
        except Exception as e:
            return jsonify({"error": f"이미지 디코딩 실패: {str(e)}"}), 400
        
        # 실제 AI 모델로 얼굴 감지 및 임베딩 추출
        detection_result = detect_faces_in_frame(cv_image)
        
        # 얼굴 데이터 추출 (detect_faces_in_frame은 { faces: [...], ... } 구조 반환)
        detected_faces = detection_result.get('faces', [])
        
        # 결과 처리
        results = {
            "total_faces": len(detected_faces),
            "faces": [],
            "criminal_detected": False,
            "normal_detected": False
        }
        
        for i, face_data in enumerate(detected_faces):
            try:
                # 얼굴 좌표 추출
                bbox = face_data.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # 기본 정보 추출
                person_name = "Unknown"
                person_type = "unknown"
                confidence = face_data.get('recognition_confidence', 0.0)
                alert_color = "yellow"  # 기본값 (미확인)
                
                # 용의자 매칭 정보 확인
                suspect_id = face_data.get('suspect_id')
                if suspect_id and confidence > 0.6:
                    # suspect_profiles에서 정보 조회
                    if suspect_id in suspect_embeddings:
                        person_data = suspect_embeddings[suspect_id]
                        person_name = person_data['name']
                        person_type = person_data['type']
                        
                        # 범죄자면 빨간색, 일반인이면 초록색
                        if person_type == "criminal":
                            alert_color = "red"
                            results["criminal_detected"] = True
                            logger.warning(f"🚨 범죄자 감지: {person_name} (신뢰도: {confidence:.3f})")
                        else:
                            alert_color = "green"
                            results["normal_detected"] = True
                            logger.info(f"✅ 일반인 감지: {person_name} (신뢰도: {confidence:.3f})")
                else:
                    # 신뢰도 낮거나 매칭 실패
                    alert_color = "yellow"
                    logger.info(f"🔍 신원 미확인 얼굴 감지 (신뢰도: {confidence:.3f})")
                
                face_result = {
                    "id": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(face_data.get('confidence', 0.9)),
                    "person_name": person_name,
                    "person_type": person_type,
                    "recognition_confidence": float(confidence),
                    "alert_color": alert_color,
                    "is_criminal": person_type == "criminal",
                    "is_normal": person_type == "normal"
                }
                
                results["faces"].append(face_result)
                
            except Exception as e:
                logger.error(f"얼굴 {i+1} 처리 중 오류: {str(e)}")
                # 오류가 발생한 얼굴도 기본 정보로 추가
                face_result = {
                    "id": i + 1,
                    "bbox": [0, 0, 100, 100],  # 기본 박스
                    "confidence": 0.5,
                    "person_name": "Unknown",
                    "person_type": "unknown",
                    "recognition_confidence": 0.0,
                    "alert_color": "gray",
                    "is_criminal": False,
                    "is_normal": False
                }
                results["faces"].append(face_result)
        
        logger.info(f"🎯 AI 감지 결과: {results['total_faces']}개 얼굴, 범죄자: {results['criminal_detected']}, 일반인: {results['normal_detected']}")
        
        response = {
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "ai_model": "InsightFace (RetinaFace + ArcFace)",
            "embeddings_loaded": len(suspect_embeddings) > 0
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"얼굴 인식 처리 중 오류: {str(e)}")
        return jsonify({
            "error": f"얼굴 인식 중 오류 발생: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/suspect/<suspect_id>', methods=['GET'])
def get_suspect_info(suspect_id):
    """특정 용의자 정보 조회"""
    try:
        if not face_detector:
            return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500
        
        # 실제 데이터베이스에서 용의자 정보 조회
        suspect_info = face_detector.get_suspect_profile(suspect_id)
        
        if suspect_info:
            return jsonify({
                "success": True,
                "suspect": suspect_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": f"용의자 ID {suspect_id}를 찾을 수 없습니다.",
                "timestamp": datetime.now().isoformat()
            }), 404
            
    except Exception as e:
        logger.error(f"용의자 정보 조회 중 오류: {str(e)}")
        return jsonify({
            "error": f"용의자 정보 조회 중 오류: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/suspects', methods=['GET'])
def get_all_suspects():
    """모든 용의자 정보 조회"""
    try:
        if not face_detector:
            return jsonify({"error": "AI 모델이 초기화되지 않았습니다."}), 500
        
        # 실제 데이터베이스에서 모든 용의자 정보 조회
        all_suspects = face_detector.get_all_suspect_profiles()
        
        return jsonify({
            "success": True,
            "suspects": all_suspects,
            "total_count": len(all_suspects),
            "timestamp": datetime.now().isoformat()
        })
            
    except Exception as e:
        logger.error(f"전체 용의자 정보 조회 중 오류: {str(e)}")
        return jsonify({
            "error": f"전체 용의자 정보 조회 중 오류: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "파일 크기가 너무 큽니다."}), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    logger.info("Starting Real AI Face Detection Server...")
    
    if initialize_face_detector():
        logger.info("🚀 Real AI Face Detection Server 시작")
        logger.info("📱 웹 인터페이스: http://localhost:5000")
        logger.info("🤖 AI 모델: InsightFace (RetinaFace + ArcFace)")
        logger.info("💻 시뮬레이션 제거 완료 - 실제 AI 모델만 사용")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        logger.error("❌ AI 시스템 초기화 실패로 서버를 시작할 수 없습니다.")
        sys.exit(1)