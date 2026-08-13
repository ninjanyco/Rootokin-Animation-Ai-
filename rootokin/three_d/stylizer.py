from pathlib import Path
from typing import Any, Dict, List

from rootokin.lattice.nodes import Embedding
from rootokin.reference_engine.embedder import embed_images


def analyze_and_stylize_3d(
    model_path: Path,
    style_prompt: str,
    style_ref_images: List[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Placeholder 3D analysis + stylization output contract."""
    _unused = (model_path, style_prompt)
    output_dir.mkdir(parents=True, exist_ok=True)
    control_maps = {
        "depth": output_dir / "depth_maps",
        "normal": output_dir / "normal_maps",
        "pose": output_dir / "pose_maps",
        "stylized_views": output_dir / "stylized",
    }
    stylized_sources = style_ref_images or [output_dir / "stylized" / "view_000.png"]
    stylized_embeds: List[Embedding] = embed_images(stylized_sources)
    return {
        "control_maps": control_maps,
        "stylized_embeds": stylized_embeds,
    }
