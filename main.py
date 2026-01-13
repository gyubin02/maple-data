#!/usr/bin/env python3
from __future__ import annotations

from contextlib import asynccontextmanager
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
CATEGORY_KEYWORDS = [
    "모자",
    "신발",
    "장갑",
    "무기",
    "상의",
    "하의",
    "망토",
    "케이프",
    "귀걸이",
    "귀고리",
    "반지",
    "목걸이",
    "벨트",
    "얼굴장식",
    "눈장식",
    "보조무기",
    "방패",
]


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
    for keyword in CATEGORY_KEYWORDS:
        if keyword in query and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def build_metadata_filter(keywords: List[str]) -> Dict[str, Any] | None:
    if not keywords:
        return None
    conditions = []
    for keyword in keywords:
        for field in ("label", "label_ko", "item_name"):
            conditions.append({field: {"$contains": keyword}})
    return {"$or": conditions}


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(DATA_DIR)), name="images")
else:
    print(f"Warning: static images directory not found: {DATA_DIR}")


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
                include=["distances", "metadatas", "ids"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Filtered query failed ({exc}); falling back to vector-only.")
            results = None

    if not results or not results.get("ids") or not results["ids"][0]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=payload.k,
            include=["distances", "metadatas", "ids"],
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
