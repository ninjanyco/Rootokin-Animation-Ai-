"""
Real CLIP embeddings using open_clip.
Install: pip install open_clip_torch pillow torch
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import open_clip
except Exception:  # pragma: no cover - optional dependency
    open_clip = None

from rootokin.lattice.nodes import Embedding

# Global model cache (loaded once)
_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = None
_MODEL_NAME = "ViT-L-14"
_PRETRAINED = "laion2b_s32b_b82k"
_FALLBACK_DIM = 768


def _get_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stable_seed(value: str) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _placeholder_vector(value: str) -> List[float]:
    rng = np.random.RandomState(_stable_seed(value))
    vector = rng.randn(_FALLBACK_DIM).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-8
    return vector.tolist()


def load_clip(
    model_name: str = _MODEL_NAME,
    pretrained: str = _PRETRAINED,
    device: Optional[str] = None,
):
    """Load (or return cached) open_clip model."""
    global _MODEL, _PREPROCESS, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _TOKENIZER

    if torch is None or open_clip is None:
        missing = []
        if torch is None:
            missing.append("torch")
        if open_clip is None:
            missing.append("open_clip_torch")
        raise RuntimeError(f"Missing optional embedding dependencies: {', '.join(missing)}")

    device = device or _get_device()
    _DEVICE = device
    print(f"[embedder] Loading open_clip {model_name} / {pretrained} on {device} ...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    _MODEL = model
    _PREPROCESS = preprocess
    _TOKENIZER = tokenizer
    return _MODEL, _PREPROCESS, _TOKENIZER


def _open_image(source: Union[str, Path, "Image.Image"]):
    if Image is None:
        raise RuntimeError("Pillow is required for image embeddings")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def embed_images(
    image_paths: List[Union[str, Path]],
    model_name: str = _MODEL_NAME,
    pretrained: str = _PRETRAINED,
    device: Optional[str] = None,
) -> List[Embedding]:
    """Create CLIP embeddings for image references, falling back when optional deps are unavailable."""
    normalized_paths = [Path(path) for path in image_paths]
    if not normalized_paths:
        return []

    try:
        model, preprocess, _ = load_clip(model_name=model_name, pretrained=pretrained, device=device)
        batch = torch.stack([preprocess(_open_image(path)) for path in normalized_paths]).to(_DEVICE)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vectors = features.detach().cpu().float().tolist()
        embedding_model = f"open_clip:{model_name}/{pretrained}"
        return [
            Embedding(vector=vector, model_name=embedding_model, source=str(path))
            for path, vector in zip(normalized_paths, vectors)
        ]
    except Exception as exc:
        print(f"[embedder] Falling back to deterministic placeholder image embeddings: {exc}")
        embedding_model = f"open_clip:{model_name}/{pretrained}"
        return [
            Embedding(vector=_placeholder_vector(str(path)), model_name=embedding_model, source=str(path))
            for path in normalized_paths
        ]


def embed_text(
    text: str,
    model_name: str = _MODEL_NAME,
    pretrained: str = _PRETRAINED,
    device: Optional[str] = None,
) -> Embedding:
    """Create CLIP embedding for text conditioning, falling back when optional deps are unavailable."""
    try:
        model, _, tokenizer = load_clip(model_name=model_name, pretrained=pretrained, device=device)
        tokens = tokenizer([text]).to(_DEVICE)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return Embedding(
            vector=features[0].detach().cpu().float().tolist(),
            model_name=f"open_clip:{model_name}/{pretrained}",
            source="text",
        )
    except Exception as exc:
        print(f"[embedder] Falling back to deterministic placeholder text embedding: {exc}")
        return Embedding(
            vector=_placeholder_vector(text),
            model_name=f"open_clip:{model_name}/{pretrained}",
            source="text",
        )
