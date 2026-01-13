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

이 프로젝트는 메이플스토리 아이템 이미지를 AI(SigLIP + LoRA)로 분석하여, 텍스트로 검색할 수 있게 해주는 백엔드 서버입니다.

- **Model:** SigLIP (LoRA Fine-tuned)
- **Database:** ChromaDB (Vector Search)
- **Framework:** FastAPI