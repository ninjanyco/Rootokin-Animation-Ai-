"""
Extract keyframes from a generated shot and push them back into the Continuity Lattice.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CV2 = False

from rootokin.lattice.continuity_lattice import LatticeManager
from rootokin.lattice.nodes import ShotNode
from rootokin.reference_engine.embedder import embed_images


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    num_keyframes: int = 3,
    max_side: int = 512,
) -> List[Path]:
    """
    Extract evenly spaced keyframes from a video.
    Returns list of saved PNG paths.
    Requires: pip install opencv-python-headless
    """
    if not HAS_CV2:
        print("[keyframe] opencv not installed – skipping real extraction")
        output_dir.mkdir(parents=True, exist_ok=True)
        dummies: List[Path] = []
        for i in range(num_keyframes):
            path = output_dir / f"kf_{i:02d}.png"
            path.touch()
            dummies.append(path)
        return dummies

    if not video_path.exists() or video_path.stat().st_size == 0:
        print(f"[keyframe] Video missing or empty: {video_path}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int)
    saved: List[Path] = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        out_path = output_dir / f"kf_{i:02d}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)

    cap.release()
    return saved


def update_lattice_with_shot(
    mgr: LatticeManager,
    shot: ShotNode,
    video_path: Path,
    keyframe_dir: Path,
    num_keyframes: int = 3,
) -> None:
    """
    1. Extract keyframes
    2. Embed them with the real CLIP embedder
    3. Attach to the ShotNode
    4. Push embeddings into the lattice memory store
    """
    kf_paths = extract_keyframes(video_path, keyframe_dir, num_keyframes=num_keyframes)
    shot.keyframe_paths = kf_paths

    if not kf_paths:
        return

    embeds = embed_images(kf_paths)
    shot.retrieved_context["keyframe_embeddings"] = [embed.model_dump() for embed in embeds]

    for embed, path in zip(embeds, kf_paths):
        mgr.add_to_memory(
            embed,
            {
                "type": "shot_keyframe",
                "shot_id": shot.id,
                "shot_index": shot.index,
                "path": str(path),
                "characters": shot.characters,
            },
        )

    print(f"[keyframe] Shot {shot.index}: {len(kf_paths)} keyframes embedded and stored in lattice")
