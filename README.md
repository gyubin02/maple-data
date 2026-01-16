---
title: Maple Story AI Search
emoji: 🍁
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# MapleStory AI Search Backend

메이플스토리 아이템 아이콘을 자연어로 검색할 수 있는 백엔드 프로젝트입니다. 아이콘 이미지를 SigLIP + LoRA로 임베딩하고, ChromaDB에 적재해 텍스트 질의로 검색합니다.

- **데모:** https://maple-data-frontend.vercel.app/
- **Model:** SigLIP (LoRA Fine-tuned)
- **Database:** ChromaDB (Vector Search)
- **Framework:** FastAPI

## 주요 기능
- 한국어 자연어 질의 → 이미지 임베딩 검색
- 라벨/태그 기반 필터링(카테고리, 색상, 분위기 키워드)
- 아이템 이미지 경로 및 메타데이터 반환

## 실행 방법
### 1) 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 데이터 수집 (선택)
Nexon Open API 키가 필요합니다.
```bash
export NEXON_API_KEY=YOUR_KEY
python -m pipeline run --date 2026-01-11 --top 100 --download-icons
```

### 3) 라벨링 (선택)
```bash
python -m labeler run \
  --input data/2026-01-11/manifest.parquet \
  --outdir data/2026-01-11/labels \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --device auto \
  --batch-size 8
```

### 4) 학습 (선택)
```bash
python train.py --data-file data/2026-01-11/labels/labels.jsonl
```

### 5) 인덱싱
```bash
python indexer.py --data-dir data/2026-01-11
```

### 6) 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

## 환경 변수
- `NEXON_API_KEY`: 데이터 수집용 Nexon Open API 키
- `OUTPUT_DIR`: 파이프라인 출력 디렉터리 (기본 `data`)
- `DB_PATH`: 파이프라인 SQLite 경로 (기본 `data/<DATE>/db.sqlite`)
- `ALLOWED_ORIGINS`: CORS 허용 도메인 목록 (쉼표 구분)

## 구성 및 흐름
1. **데이터 수집:** Nexon Open API로 랭킹/아이템/아이콘 수집 (`src/pipeline`)
2. **라벨링:** Qwen2-VL로 아이콘 텍스트 라벨 생성 (`labeler`, `src/labeler`)
3. **학습:** SigLIP에 LoRA 파인튜닝 (`train.py`)
4. **인덱싱:** 이미지 임베딩을 ChromaDB에 적재 (`indexer.py`)
5. **서비스:** 검색 API 제공 (`main.py`)

## API
- `POST /search`
  - 입력: `{ "query": "파란 모자", "k": 10 }`
  - 출력: 유사도와 메타데이터가 포함된 결과 목록

## 참고 문서
- 라벨링 파이프라인 상세: `README_labeling.md`
