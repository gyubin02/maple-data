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
    collection = client.get_or_create_collection(name="maple_items")

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
        if metadata:
            filepath = metadata.get("filepath", "")
        image_url = f"/static/images/{filepath}" if filepath else ""
        response_items.append(
            {
                "id": item_id,
                "filepath": filepath,
                "distance": distance,
                "image_url": image_url,
            }
        )

    return {
        "query": query,
        "k": payload.k,
        "results": response_items,
    }
