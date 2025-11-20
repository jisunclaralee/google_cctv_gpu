# CCTV Suspect Identification System

Flask 기반의 AI 파이프라인으로 CCTV 영상에서 얼굴을 탐지하고, 등록된 임베딩과 비교해 용의자를 식별하는 프로젝트입니다.  
RetinaFace + ArcFace(InsightFace)를 사용해 얼굴을 검출/인식하며, 템플릿(`templates/index.html`)에서 비디오 업로드·상태판을 표시합니다.

---

## 주요 기능

1. **실시간 얼굴 검출 및 식별**  
   - `models/retinaface.py`, `models/arcface.py`, `models/real_face_detector.py` 에서 InsightFace 기반 모델 로직 제공.
2. **Flask REST API**  
   - `api/server_real.py`가 주요 엔드포인트(`/api/status`, `/api/detect` 등)를 제공하고, 프론트엔드 템플릿도 서빙.
3. **정교한 프론트엔드 대시보드**  
   - `templates/index.html`은 비디오 업로드, 디버그 오버레이, 감지 로그를 한 번에 확인할 수 있는 Tailwind UI.
4. **데이터/임베딩 관리**  
   - `data/embeddings/`의 JSON 파일로 등록 얼굴의 벡터를 관리하고, `data/suspects`에는 메타데이터·이미지를 저장.
5. **서비스 파이프라인 및 설정 분리**  
   - `services/face_pipeline.py`는 검출→전처리→식별 흐름을 담당하고, `config.py`에서 공통 설정을 관리.

---

## 디렉터리 구조

```
google_cctv-jisun/
├── api/                   # Flask 서버
│   ├── server_real.py     # 실제 감지 서비스 엔드포인트
│   └── server.py          # 초기/테스트용 서버
├── models/
│   ├── retinaface.py      # 얼굴 검출 모듈
│   ├── arcface.py         # 얼굴 인식 모듈
│   └── real_face_detector.py
├── services/
│   └── face_pipeline.py   # 검출→식별 서비스 로직
├── templates/
│   └── index.html         # 단일 페이지 UI
├── data/
│   ├── embeddings/        # 얼굴 임베딩 JSON
│   └── suspects/          # 프로필 및 원본 이미지
├── uploads/               # 업로드된 비디오가 저장되는 임시 폴더
├── config.py              # 공통 설정 (GPU 사용 여부 등)
├── requirements.txt       # 파이썬 의존성 목록
├── install.py             # 의존성/ONNX 모델 설치 스크립트
├── run_gpu.bat            # Windows용 실행 배치
└── README.md              # (현재 문서)
```

> `cctv_suspect_identification.html`, `temp_script.txt` 등 과거 테스트 파일은 제거했습니다.  
> `logs/`, `uploads/` 는 실행 중 생성되는 임시 데이터 폴더입니다.

---

## 설치 및 실행

### 1) 환경 준비
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell 기준
pip install --upgrade pip
pip install -r requirements.txt
```

> GPU 사용 시 PyTorch CUDA 패키지가 requirements에 포함되어 있습니다. CUDA Toolkit 11/12가 설치되어 있어야 합니다.

### 2) 모델/임베딩 준비

1. `install.py` 실행 시 ONNX 모델과 기본 임베딩을 자동 다운로드하도록 구성되어 있습니다.
   ```bash
   python install.py
   ```
2. 추가 인물을 등록하려면 `data/suspects/metadata/suspect_profiles.json`과 `data/embeddings/`를 업데이트하세요.

### 3) 서버 실행

```bash
python api/server_real.py
```

웹 브라우저에서 `http://localhost:5000` 에 접속해 UI를 확인합니다.

---

## 주요 파일 설명

| 경로 | 설명 |
| --- | --- |
| `api/server_real.py` | Flask 앱 진입점. `/api/status`, `/api/detect`, `/api/suspects` 등 엔드포인트 정의 및 템플릿 렌더링 |
| `models/real_face_detector.py` | InsightFace를 초기화하고, 임베딩/프로필 데이터를 읽어 얼굴 감지 결과를 반환 |
| `services/face_pipeline.py` | 단일 프레임을 받아 검출→정규화→인식까지 수행하는 서비스 레이어 |
| `templates/index.html` | 업로드→감지→오버레이 표시·디버그 로그까지 모두 포함하는 SPA 형태의 UI |
| `config.py` | GPU/CPU 여부, 임계값, 모델 경로 등 공통 설정 |
| `requirements.txt` | Flask, InsightFace, onnxruntime, numpy 등 필요한 파이썬 패키지 명시 |

---

## 커밋/배포 가이드

1. **정리**: 로그/업로드/모델 캐시 등 대용량 임시 파일은 `.gitignore` 에 포함하거나 커밋 전 비우세요.
2. **테스트**: `python api/server_real.py` 를 실행 후 비디오 업로드 → 모델 연결 테스트 → 프레임 분석이 정상인지 확인.
3. **Git 커밋 예시**
   ```bash
   git add .
   git commit -m "feat: integrate real-time suspect dashboard"
   git push origin main
   ```

---

## FAQ

- **Q. 비디오가 회색 화면만 나온다?**  
  → 파일 업로드 후 `현상수배범 감지 필터`를 활성화하고, 브라우저 콘솔 오류(`URL.createObjectURL` 등)가 없는지 확인하세요.

- **Q. 모델 연결이 안 된다?**  
  → `api/server_real.py` 로그에서 InsightFace 초기화/임베딩 로드를 확인하고, `data/embeddings/*.json` 경로가 올바른지 체크하세요.

---

필요한 항목이나 수정 요청이 있으면 README 상단 내용을 참고해 빠르게 대응할 수 있습니다. 즐거운 개발 되세요! 🎯
