from pathlib import Path
from typing import Any, Dict

from rootokin.lattice.nodes import ContinuityLattice, ShotNode


class RootokinGenerator:
    """
    Ensemble wrapper.
    Preferred order for long-form consistency (as of Aug 2026):
    1. SkyReels-V3  (best multi-ref + extension)
    2. LongCat-Video (native minutes-long continuation, MIT)
    3. LTX-2.3      (highest local quality + native audio)
    4. Wan 2.x      (strong open r2v / character consistency)
    """

    def __init__(self, preferred_backend: str = "skyreels"):
        self.backend = preferred_backend
        self.pipelines: Dict[str, Any] = {}

    def condition_from_lattice(self, lattice: ContinuityLattice, shot: ShotNode) -> Dict[str, Any]:
        ctx = lattice.retrieve_for_shot(shot)
        prompt = self._build_prompt(ctx, shot)
        ref_images = []
        control_maps = {}

        for char in ctx["characters"].values():
            ref_images.extend(char.reference_images)
            control_maps.update(char.control_maps)

        return {
            "prompt": prompt,
            "ref_images": ref_images,
            "control_maps": control_maps,
            "style_prompt": ctx["style"].prompt,
            "previous_keyframes": ctx["previous_keyframes"],
            "physics_constraints": ctx["physics"],
        }

    def _build_prompt(self, ctx: Dict[str, Any], shot: ShotNode) -> str:
        chars = ", ".join(character.name for character in ctx["characters"].values())
        return (
            f"{shot.script_beat}. "
            f"Characters: {chars}. "
            f"Style: {ctx['style'].prompt}. "
            f"World rules: {ctx['world'].physics_rules}. "
            "Maintain exact continuity with previous frames."
        )

    def generate_shot(self, lattice: ContinuityLattice, shot: ShotNode, output_dir: Path) -> Path:
        self.condition_from_lattice(lattice, shot)
        video_path = output_dir / f"shot_{shot.index:04d}.mp4"
        shot.generated_video_path = video_path
        return video_path
