from pathlib import Path
from typing import List

import numpy as np

from rootokin.lattice.nodes import Embedding


def embed_images(image_paths: List[Path], model_name: str = "clip-vit-large") -> List[Embedding]:
    """Create deterministic placeholder embeddings for image references."""
    embeddings: List[Embedding] = []
    for image_path in image_paths:
        rng = np.random.RandomState(abs(hash(str(image_path))) % (2**32))
        vector = rng.randn(768).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-8
        embeddings.append(
            Embedding(
                vector=vector.tolist(),
                model_name=model_name,
                source=str(image_path),
            )
        )
    return embeddings


def embed_text(text: str, model_name: str = "clip-vit-large") -> Embedding:
    """Create deterministic placeholder embedding for text conditioning."""
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    vector = rng.randn(768).astype(np.float32)
    vector /= np.linalg.norm(vector) + 1e-8
    return Embedding(vector=vector.tolist(), model_name=model_name, source="text")
