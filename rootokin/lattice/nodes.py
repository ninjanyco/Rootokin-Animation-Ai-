from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, PrivateAttr


class Embedding(BaseModel):
    vector: List[float]  # CLIP / model-specific embedding
    model_name: str = "clip-vit-large"
    source: str = ""  # "image", "3d_render", "keyframe", etc.


class CharacterNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    reference_images: List[Path] = Field(default_factory=list)
    embeddings: List[Embedding] = Field(default_factory=list)
    three_d_path: Optional[Path] = None
    control_maps: Dict[str, Path] = Field(default_factory=dict)  # depth/pose/normal/stylized views
    behavioral_traits: Dict[str, Any] = Field(default_factory=dict)
    appearance_history: List[Embedding] = Field(default_factory=list)  # evolving but constrained


class StyleNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    prompt: str
    palette: List[str] = Field(default_factory=list)
    reference_images: List[Path] = Field(default_factory=list)
    embeddings: List[Embedding] = Field(default_factory=list)
    rendering_mode: str = "cinematic"  # or "anime", "painterly", etc.


class WorldNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    physics_rules: Dict[str, Any] = Field(default_factory=dict)  # gravity, lighting model, constraints
    architecture: str = ""
    symbolic_logic: List[str] = Field(default_factory=list)
    global_embeddings: List[Embedding] = Field(default_factory=list)


class ShotNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    index: int
    start_time: float  # seconds
    duration: float
    script_beat: str
    characters: List[str]  # character ids
    camera_cues: str = ""
    emotional_arc: str = ""
    previous_shot_id: Optional[str] = None
    keyframe_paths: List[Path] = Field(default_factory=list)
    generated_video_path: Optional[Path] = None
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)  # what lattice gave this shot


class ContinuityLattice(BaseModel):
    world: WorldNode
    style: StyleNode
    characters: Dict[str, CharacterNode] = Field(default_factory=dict)
    shots: List[ShotNode] = Field(default_factory=list)
    memory_index: Dict[str, Any] = Field(default_factory=dict)  # later FAISS/Chroma
    _shot_index: Dict[str, ShotNode] = PrivateAttr(default_factory=dict)

    def add_character(self, char: CharacterNode) -> None:
        self.characters[char.id] = char

    def inject_image_refs(self, character_id: str, image_paths: List[Path], embeddings: List[Embedding]) -> None:
        char = self.characters[character_id]
        char.reference_images.extend(image_paths)
        char.embeddings.extend(embeddings)

    def inject_3d(
        self,
        character_id: str,
        model_path: Path,
        control_maps: Dict[str, Path],
        stylized_embeds: List[Embedding],
    ) -> None:
        char = self.characters[character_id]
        char.three_d_path = model_path
        char.control_maps.update(control_maps)
        char.embeddings.extend(stylized_embeds)

    def refresh_indexes(self) -> None:
        self._shot_index = {shot.id: shot for shot in self.shots}

    def retrieve_for_shot(self, shot: ShotNode, top_k: int = 4) -> Dict[str, Any]:
        """Semantic + temporal retrieval for conditioning."""
        del top_k  # placeholder for future ranker
        relevant_chars = {cid: self.characters[cid] for cid in shot.characters if cid in self.characters}
        prev_keyframes: List[Path] = []
        if shot.previous_shot_id:
            if shot.previous_shot_id not in self._shot_index:
                self.refresh_indexes()
            prev_shot = self._shot_index.get(shot.previous_shot_id)
            if prev_shot:
                prev_keyframes = prev_shot.keyframe_paths

        return {
            "world": self.world,
            "style": self.style,
            "characters": relevant_chars,
            "previous_keyframes": prev_keyframes,
            "physics": self.world.physics_rules,
            "semantic_memory": [],
        }
