#!/usr/bin/env python3
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import SiglipModel, SiglipProcessor


DATA_DIR = (Path(__file__).resolve().parent / "data/2026-01-11").resolve()
CATEGORY_SYNONYMS = {
    "모자": ["모자", "헬름", "헬멧", "햇", "보닛", "캡"],
    "신발": ["신발", "슈즈", "부츠", "샌들"],
    "장갑": ["장갑", "글러브"],
    "무기": ["무기", "검", "소드", "대검", "스태프", "완드", "활", "석궁", "창", "스피어", "폴암", "도끼", "단검", "너클", "건", "총", "클로"],
    "상의": ["상의", "셔츠", "자켓", "코트", "로브", "블라우스"],
    "하의": ["하의", "바지", "팬츠", "스커트"],
    "망토": ["망토", "케이프", "cape"],
    "귀걸이": ["귀걸이", "귀고리", "이어링"],
    "반지": ["반지", "링"],
    "목걸이": ["목걸이", "펜던트", "네클리스"],
    "벨트": ["벨트"],
    "얼굴장식": ["얼굴장식", "얼굴 장식"],
    "눈장식": ["눈장식", "눈 장식"],
    "보조무기": ["보조무기", "보조 무기"],
    "방패": ["방패", "쉴드", "실드"],
}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=100)


def resolve_adapter_path(adapter_path: Path) -> Path:
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    candidate = adapter_path / "best_model"
    if (candidate / "adapter_config.json").exists():
        return candidate
    return adapter_path


def extract_category_keywords(query: str) -> List[str]:
    keywords: List[str] = []
    lowered_query = query.lower()
    for category, variants in CATEGORY_SYNONYMS.items():
        for variant in variants:
            if variant.lower() in lowered_query and category not in keywords:
                keywords.append(category)
                break
    return keywords


def build_metadata_filter(keywords: List[str]) -> Dict[str, Any] | None:
    if not keywords:
        return None
    return {"category": {"$in": keywords}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_id = "google/siglip-base-patch16-256-multilingual"
    adapter_path = resolve_adapter_path(Path("outputs/ko-clip-lora"))

    print("Loading SigLIP + LoRA model...")
    base_model = SiglipModel.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    processor = SiglipProcessor.from_pretrained(base_model_id)

    model.to(device)
    model.eval()

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="maple_items",
        metadata={"hnsw:space": "cosine"},
    )

    app.state.device = device
    app.state.model = model
    app.state.processor = processor
    app.state.collection = collection

    yield


app = FastAPI(lifespan=lifespan)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [
        origin.strip()
        for origin in allowed_origins_env.split(",")
        if origin.strip()
    ]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(DATA_DIR)), name="images")
else:
    print(f"Warning: static images directory not found: {DATA_DIR}")


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(payload: SearchRequest) -> Dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    model: SiglipModel = app.state.model
    processor: SiglipProcessor = app.state.processor
    device: torch.device = app.state.device
    collection = app.state.collection

    with torch.inference_mode():
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)

    query_embedding = text_embeds[0].detach().cpu().tolist()

    filter_keywords = extract_category_keywords(query)
    where_filter = build_metadata_filter(filter_keywords)

    results = None
    if where_filter:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=payload.k,
                where=where_filter,
                include=["distances", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Filtered query failed ({exc}); falling back to vector-only.")
            results = None

    if not results or not results.get("ids") or not results["ids"][0]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=payload.k,
            include=["distances", "metadatas"],
        )

    ids: List[str] = results.get("ids", [[]])[0]
    distances: List[float] = results.get("distances", [[]])[0]
    metadatas: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]

    response_items = []
    for item_id, distance, metadata in zip(ids, distances, metadatas):
        filepath = ""
        item_name = ""
        label_ko = ""
        if metadata:
            filepath = metadata.get("filepath", "")
            item_name = metadata.get("item_name", "") or ""
            label_ko = metadata.get("label_ko") or metadata.get("label") or ""
        if not item_name and filepath:
            item_name = Path(filepath).stem
        image_url = f"/static/images/{filepath}" if filepath else ""
        similarity = max(0.0, 1.0 - distance) if distance is not None else 0.0
        response_items.append(
            {
                "id": item_id,
                "filepath": filepath,
                "distance": distance,
                "similarity": similarity,
                "image_url": image_url,
                "item_name": item_name,
                "label_ko": label_ko,
                "label": label_ko,
            }
        )

    return {
        "query": query,
        "k": payload.k,
        "results": response_items,
    }
