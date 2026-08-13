from pathlib import Path
from typing import Any, Dict


def load_3d_model(model_path: Path) -> Dict[str, Any]:
    """Basic analysis stub for 3D model loading."""
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        "path": str(model_path),
        "format": model_path.suffix.lower(),
        "has_skeleton": True,
        "mesh_count": 1,
        "bone_names": ["root", "spine", "head"],
    }
