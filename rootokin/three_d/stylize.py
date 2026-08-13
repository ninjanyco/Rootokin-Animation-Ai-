from pathlib import Path
from typing import Dict, List


def analyze_and_stylize_3d(
    model_path: Path,
    style_prompt: str,
    style_ref_images: List[Path],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    1. Load FBX/GLTF
    2. Render multi-view depth, normal, OpenPose-like maps
    3. Run style transfer (ControlNet depth + IP-Adapter + style prompt)
    4. Return control map paths + stylized view paths for lattice injection
    """
    del model_path, style_prompt, style_ref_images
    return {
        "depth": output_dir / "depth_maps",
        "normal": output_dir / "normal_maps",
        "pose": output_dir / "pose_maps",
        "stylized_views": output_dir / "stylized",
    }
