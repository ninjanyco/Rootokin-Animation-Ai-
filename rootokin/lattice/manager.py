from pathlib import Path
from typing import List, Optional
import json

from .nodes import ContinuityLattice, ShotNode, StyleNode, WorldNode


class LatticeManager:
    def __init__(self, lattice: Optional[ContinuityLattice] = None):
        self.lattice = lattice or ContinuityLattice(
            world=WorldNode(name="default_world"),
            style=StyleNode(name="default_style", prompt="cinematic continuity"),
        )

    def build_from_script(self, script_beats: List[dict], duration_minutes: float = 30.0) -> ContinuityLattice:
        """Turn script into ordered ShotNodes with temporal links."""
        self.lattice.shots = []

        total_seconds = duration_minutes * 60
        n = len(script_beats)
        dur = total_seconds / max(n, 1)
        prev_id = None

        for i, beat in enumerate(script_beats):
            shot = ShotNode(
                index=i,
                start_time=i * dur,
                duration=dur,
                script_beat=beat.get("text", ""),
                characters=beat.get("characters", []),
                previous_shot_id=prev_id,
            )
            self.lattice.shots.append(shot)
            prev_id = shot.id

        self.lattice.refresh_indexes()
        return self.lattice

    def save(self, path: Path) -> None:
        path.write_text(self.lattice.model_dump_json(indent=2))

    def load(self, path: Path) -> ContinuityLattice:
        data = json.loads(path.read_text())
        self.lattice = ContinuityLattice.model_validate(data)
        self.lattice.refresh_indexes()
        return self.lattice
