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

from keyword_filters import (
    CATEGORY_SYNONYMS,
    COLOR_SYNONYMS,
    VIBE_SYNONYMS,
    extract_keywords,
)

DATA_DIR = (Path(__file__).resolve().parent / "data/2026-01-11").resolve()


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


def extract_query_filters(query: str) -> Dict[str, List[str]]:
    texts = [query]
    return {
        "categories": extract_keywords(texts, CATEGORY_SYNONYMS),
        "colors": extract_keywords(texts, COLOR_SYNONYMS),
        "vibes": extract_keywords(texts, VIBE_SYNONYMS),
    }


def build_where_filter(
    categories: List[str], colors: List[str], vibes: List[str]
) -> Dict[str, Any] | None:
    clauses: List[Dict[str, Any]] = []
    if categories:
        clauses.append({"category": {"$in": categories}})
    if colors:
        clauses.append({"$and": [{f"color_{color}": True} for color in colors]})
    if vibes:
        clauses.append({"$and": [{f"vibe_{vibe}": True} for vibe in vibes]})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_filter_candidates(filters: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    parts = {
        "category": filters.get("categories") or [],
        "color": filters.get("colors") or [],
        "vibe": filters.get("vibes") or [],
    }
    candidates: List[Dict[str, Any]] = []
    combos = [
        ("category", "color", "vibe"),
        ("category", "color"),
        ("category", "vibe"),
        ("color", "vibe"),
        ("category",),
        ("color",),
        ("vibe",),
    ]
    for combo in combos:
        if not all(parts[facet] for facet in combo):
            continue
        where_filter = build_where_filter(
            parts["category"] if "category" in combo else [],
            parts["color"] if "color" in combo else [],
            parts["vibe"] if "vibe" in combo else [],
        )
        if where_filter:
            candidates.append(where_filter)
    return candidates


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

    filter_parts = extract_query_filters(query)
    where_candidates = build_filter_candidates(filter_parts)

    results = None
    for where_filter in where_candidates:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=payload.k,
                where=where_filter,
                include=["distances", "metadatas"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Filtered query failed ({exc}); trying less strict.")
            results = None
            continue
        if results and results.get("ids") and results["ids"][0]:
            break

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
