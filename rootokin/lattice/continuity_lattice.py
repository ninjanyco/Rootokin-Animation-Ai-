from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np

from .nodes import CharacterNode, ContinuityLattice, Embedding, ShotNode, StyleNode, WorldNode


class LatticeManager:
    def __init__(self, lattice: Optional[ContinuityLattice] = None):
        self.lattice = lattice or ContinuityLattice(
            world=WorldNode(name="default_world"),
            style=StyleNode(name="default_style", prompt="cinematic, continuous, coherent world"),
        )
        self._vectors: List[np.ndarray] = []
        self._meta: List[Dict[str, Any]] = []

    def _add_to_memory(self, embedding: Embedding, meta: Dict[str, Any]) -> None:
        vector = np.array(embedding.vector, dtype=np.float32)
        self._vectors.append(vector)
        self._meta.append(meta)

    def _get_previous_keyframes(self, shot: ShotNode) -> List[Path]:
        if not shot.previous_shot_id:
            return []
        for existing_shot in self.lattice.shots:
            if existing_shot.id == shot.previous_shot_id:
                return existing_shot.keyframe_paths
        return []

    def add_character(self, char: CharacterNode) -> None:
        self.lattice.add_character(char)

    def inject_image_refs(
        self,
        character_id: str,
        image_paths: List[Path],
        embeddings: List[Embedding],
    ) -> None:
        if character_id not in self.lattice.characters:
            raise KeyError(f"Character {character_id} not found")
        char = self.lattice.characters[character_id]
        char.reference_images.extend(image_paths)
        char.embeddings.extend(embeddings)
        for embedding, image_path in zip(embeddings, image_paths):
            self._add_to_memory(
                embedding,
                {
                    "type": "image_ref",
                    "character_id": character_id,
                    "path": str(image_path),
                },
            )

    def inject_3d(
        self,
        character_id: str,
        model_path: Path,
        control_maps: Dict[str, Path],
        stylized_embeds: List[Embedding],
    ) -> None:
        if character_id not in self.lattice.characters:
            raise KeyError(f"Character {character_id} not found")
        char = self.lattice.characters[character_id]
        char.three_d_path = model_path
        char.control_maps.update(control_maps)
        char.embeddings.extend(stylized_embeds)
        for embedding in stylized_embeds:
            self._add_to_memory(
                embedding,
                {
                    "type": "3d_stylized",
                    "character_id": character_id,
                    "model": str(model_path),
                },
            )

    def retrieve_for_shot(self, shot: ShotNode, top_k: int = 6) -> Dict[str, Any]:
        """Hybrid temporal + semantic retrieval."""
        relevant_chars = {
            char_id: self.lattice.characters[char_id]
            for char_id in shot.characters
            if char_id in self.lattice.characters
        }

        semantic_hits: List[Dict[str, Any]] = []
        if self._vectors and relevant_chars:
            query_vectors: List[np.ndarray] = []
            for char in relevant_chars.values():
                for embedding in char.embeddings:
                    query_vectors.append(np.array(embedding.vector, dtype=np.float32))

            if query_vectors:
                query = np.mean(query_vectors, axis=0)
                query = query / (np.linalg.norm(query) + 1e-8)
                sims: List[tuple[float, int]] = []
                for idx, vector in enumerate(self._vectors):
                    norm_vec = vector / (np.linalg.norm(vector) + 1e-8)
                    sims.append((float(np.dot(query, norm_vec)), idx))
                sims.sort(reverse=True)
                for score, idx in sims[:top_k]:
                    semantic_hits.append({"score": score, **self._meta[idx]})

        return {
            "world": self.lattice.world,
            "style": self.lattice.style,
            "characters": relevant_chars,
            "previous_keyframes": self._get_previous_keyframes(shot),
            "physics": self.lattice.world.physics_rules,
            "semantic_memory": semantic_hits,
        }

    def build_from_script(self, script_beats: List[dict], duration_minutes: float = 30.0) -> ContinuityLattice:
        total_seconds = duration_minutes * 60.0
        beat_count = max(len(script_beats), 1)
        shot_duration = total_seconds / beat_count
        previous_id = None
        self.lattice.shots = []

        for index, beat in enumerate(script_beats):
            shot = ShotNode(
                index=index,
                start_time=index * shot_duration,
                duration=shot_duration,
                script_beat=beat.get("text", ""),
                characters=beat.get("characters", []),
                camera_cues=beat.get("camera", ""),
                emotional_arc=beat.get("emotion", ""),
                previous_shot_id=previous_id,
            )
            self.lattice.shots.append(shot)
            previous_id = shot.id
        self.lattice.refresh_indexes()
        return self.lattice

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.lattice.model_dump_json(indent=2))

    def load(self, path: Path) -> ContinuityLattice:
        data = json.loads(path.read_text())
        self.lattice = ContinuityLattice.model_validate(data)
        self.lattice.refresh_indexes()
        self._vectors = []
        self._meta = []
        for char in self.lattice.characters.values():
            for embedding in char.embeddings:
                self._add_to_memory(
                    embedding,
                    {
                        "type": "restored",
                        "character_id": char.id,
                    },
                )
        return self.lattice
