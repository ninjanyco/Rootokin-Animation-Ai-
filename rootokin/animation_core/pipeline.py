from pathlib import Path
from typing import Dict, List, Optional

from rootokin.animation_core.keyframe import update_lattice_with_shot
from rootokin.lattice.continuity_lattice import LatticeManager
from rootokin.models.generator import RootokinGenerator
from rootokin.script_engine.parser import parse_script


def run_pipeline(
    lattice_path: Path,
    script_path: Path,
    duration_minutes: float,
    output_dir: Path,
    backend: str = "skyreels",
    known_characters: Optional[List[str]] = None,
) -> LatticeManager:
    manager = LatticeManager()
    if lattice_path.exists():
        manager.load(lattice_path)

    script_beats = parse_script(
        script_path,
        known_characters=known_characters or [char.name for char in manager.lattice.characters.values()],
    )
    name_to_id: Dict[str, str] = {char.name.lower(): char.id for char in manager.lattice.characters.values()}
    for beat in script_beats:
        beat["characters"] = [
            name_to_id.get(character.lower(), character)
            for character in beat.get("characters", [])
            if character != "main"
        ]
    manager.build_from_script(script_beats=script_beats, duration_minutes=duration_minutes)

    generator = RootokinGenerator(preferred_backend=backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.load()

    for shot in manager.lattice.shots:
        print(f"\n=== Shot {shot.index} ===")
        print(f"Beat: {shot.script_beat[:80]}...")
        generated_path = generator.generate_shot(
            manager,
            shot,
            output_dir,
            duration_sec=shot.duration,
        )
        keyframe_dir = output_dir / f"keyframes_shot_{shot.index:04d}"
        update_lattice_with_shot(manager, shot, generated_path, keyframe_dir)
        manager.lattice.memory_index[f"shot_{shot.index:04d}"] = {
            "video_path": str(generated_path),
            "context": shot.retrieved_context,
            "keyframes": [str(path) for path in shot.keyframe_paths],
        }

    manager.save(lattice_path)
    manager.save(output_dir / "lattice_final.json")
    print(f"\nPipeline complete. Final lattice → {output_dir / 'lattice_final.json'}")
    return manager
