from pathlib import Path
from typing import Any, Dict

from rootokin.lattice.nodes import ContinuityLattice, ShotNode


class RootokinGenerator:
    """Lightweight ensemble wrapper placeholder."""

    def __init__(self, preferred_backend: str = "skyreels"):
        self.backend = preferred_backend
        self.pipelines: Dict[str, Any] = {}

    def condition_from_lattice(self, lattice: ContinuityLattice, shot: ShotNode) -> Dict[str, Any]:
        if hasattr(lattice, "retrieve_for_shot"):
            context = lattice.retrieve_for_shot(shot)
        else:
            context = {
                "world": lattice.world,
                "style": lattice.style,
                "characters": {cid: lattice.characters[cid] for cid in shot.characters if cid in lattice.characters},
                "previous_keyframes": [],
                "physics": lattice.world.physics_rules,
                "semantic_memory": [],
            }

        prompt = self._build_prompt(context, shot)
        ref_images = []
        control_maps = {}
        for char in context["characters"].values():
            ref_images.extend(char.reference_images)
            control_maps.update(char.control_maps)

        return {
            "prompt": prompt,
            "ref_images": ref_images,
            "control_maps": control_maps,
            "style_prompt": context["style"].prompt,
            "previous_keyframes": context["previous_keyframes"],
            "physics_constraints": context["physics"],
            "semantic_memory": context.get("semantic_memory", []),
        }

    def _build_prompt(self, context: Dict[str, Any], shot: ShotNode) -> str:
        char_names = ", ".join(character.name for character in context["characters"].values())
        return (
            f"{shot.script_beat}. "
            f"Characters: {char_names}. "
            f"Style: {context['style'].prompt}. "
            f"World rules: {context['world'].physics_rules}. "
            "Maintain exact continuity with previous frames."
        )

    def generate_shot(self, lattice: ContinuityLattice, shot: ShotNode, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        shot.retrieved_context = self.condition_from_lattice(lattice, shot)
        video_path = output_dir / f"shot_{shot.index:04d}.mp4"
        shot.generated_video_path = video_path
        return video_path
