from pathlib import Path
from typing import List
import json

from rootokin.lattice.continuity_lattice import LatticeManager
from rootokin.models.generator import RootokinGenerator


def _read_script(script_path: Path) -> List[dict]:
    text = script_path.read_text().strip()
    if not text:
        return []
    if script_path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "beats" in data and isinstance(data["beats"], list):
            return data["beats"]
        raise ValueError("Unsupported script JSON format")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"text": line, "characters": []} for line in lines]


def run_pipeline(
    lattice_path: Path,
    script_path: Path,
    duration_minutes: float,
    output_dir: Path,
    backend: str = "skyreels",
) -> LatticeManager:
    manager = LatticeManager()
    if lattice_path.exists():
        manager.load(lattice_path)

    script_beats = _read_script(script_path)
    manager.build_from_script(script_beats=script_beats, duration_minutes=duration_minutes)

    generator = RootokinGenerator(preferred_backend=backend)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for shot in manager.lattice.shots:
        generated_path = generator.generate_shot(manager.lattice, shot, videos_dir)
        manager.lattice.memory_index[f"shot_{shot.index:04d}"] = {
            "video_path": str(generated_path),
            "context": shot.retrieved_context,
        }

    manager.save(lattice_path)
    return manager
