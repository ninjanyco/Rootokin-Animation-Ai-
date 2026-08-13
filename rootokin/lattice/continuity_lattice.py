from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_FAISS = False

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CHROMA = False

from .nodes import CharacterNode, ContinuityLattice, Embedding, ShotNode, StyleNode, WorldNode


class LatticeManager:
    def __init__(
        self,
        lattice: Optional[ContinuityLattice] = None,
        vector_backend: str = "faiss",
        persist_dir: Optional[Path] = None,
    ):
        self.lattice = lattice or ContinuityLattice(
            world=WorldNode(name="default_world"),
            style=StyleNode(name="default_style", prompt="cinematic, continuous, coherent world"),
        )
        self.vector_backend = vector_backend.lower()
        self.persist_dir = Path(persist_dir) if persist_dir else None

        self._index = None
        self._meta: List[Dict[str, Any]] = []
        self._dim = 768
        self._vectors: List[np.ndarray] = []

        self._init_vector_store()

    def _init_vector_store(self) -> None:
        self._index = None
        if self.vector_backend == "faiss" and HAS_FAISS:
            self._index = faiss.IndexFlatIP(self._dim)
            print("[lattice] Using FAISS (IndexFlatIP)")
        elif self.vector_backend == "chroma" and HAS_CHROMA:
            client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma") if self.persist_dir else "./chroma_db",
                settings=Settings(anonymized_telemetry=False),
            )
            self._index = client.get_or_create_collection(
                name="rootokin_memory",
                metadata={"hnsw:space": "cosine"},
            )
            existing_ids = self._index.get(include=[])["ids"]
            if existing_ids:
                self._index.delete(ids=existing_ids)
            print("[lattice] Using Chroma")
        else:
            print("[lattice] Falling back to pure NumPy (install faiss-cpu or chromadb for better scaling)")
        self._vectors = []

    def _ensure_vector_store_dim(self, vec: np.ndarray) -> None:
        if vec.shape[0] == self._dim:
            return
        has_items = bool(self._meta or self._vectors)
        if self.vector_backend == "faiss" and HAS_FAISS and self._index is not None:
            has_items = has_items or self._index.ntotal > 0
        if has_items:
            raise ValueError(f"Embedding dimension {vec.shape[0]} does not match memory dimension {self._dim}")
        self._dim = int(vec.shape[0])
        self._init_vector_store()

    def _add_to_memory(self, embedding: Embedding, meta: Dict[str, Any]) -> None:
        vec = np.array(embedding.vector, dtype=np.float32)
        self._ensure_vector_store_dim(vec)
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm

        meta = {**meta, "model": embedding.model_name, "source": embedding.source}

        if self.vector_backend == "faiss" and HAS_FAISS and self._index is not None:
            self._index.add(vec.reshape(1, -1))
            self._meta.append(meta)
        elif self.vector_backend == "chroma" and HAS_CHROMA and self._index is not None:
            uid = f"{meta.get('type', 'unk')}_{len(self._meta)}_{hash(str(meta)) % 10**8}"
            self._index.add(
                embeddings=[vec.tolist()],
                documents=[meta.get("source", "")],
                metadatas=[meta],
                ids=[uid],
            )
            self._meta.append(meta)
        else:
            self._vectors.append(vec)
            self._meta.append(meta)

    def add_to_memory(self, embedding: Embedding, meta: Dict[str, Any]) -> None:
        self._add_to_memory(embedding, meta)

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

    def retrieve_for_shot(self, shot: ShotNode, top_k: int = 8) -> Dict[str, Any]:
        """Hybrid temporal + semantic retrieval using the upgraded store."""
        relevant_chars = {
            char_id: self.lattice.characters[char_id]
            for char_id in shot.characters
            if char_id in self.lattice.characters
        }

        prev_keyframes = self._get_previous_keyframes(shot)
        semantic_hits: List[Dict[str, Any]] = []
        query_vec = None
        if relevant_chars:
            query_vecs: List[np.ndarray] = []
            for char in relevant_chars.values():
                for embedding in char.embeddings[-4:]:
                    vec = np.array(embedding.vector, dtype=np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    query_vecs.append(vec)
            if query_vecs:
                query_vec = np.mean(query_vecs, axis=0)
                query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        if query_vec is not None:
            if self.vector_backend == "faiss" and HAS_FAISS and self._index is not None and self._index.ntotal > 0:
                scores, indices = self._index.search(query_vec.reshape(1, -1), min(top_k, self._index.ntotal))
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        semantic_hits.append({"score": float(score), **self._meta[idx]})
            elif self.vector_backend == "chroma" and HAS_CHROMA and self._index is not None:
                results = self._index.query(
                    query_embeddings=[query_vec.tolist()],
                    n_results=top_k,
                    include=["metadatas", "distances"],
                )
                for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                    semantic_hits.append({"score": 1.0 - dist, **meta})
            elif self._vectors:
                sims: List[Tuple[float, int]] = []
                for idx, vector in enumerate(self._vectors):
                    sims.append((float(np.dot(query_vec, vector)), idx))
                sims.sort(reverse=True)
                for score, idx in sims[:top_k]:
                    semantic_hits.append({"score": score, **self._meta[idx]})

        return {
            "world": self.lattice.world,
            "style": self.lattice.style,
            "characters": relevant_chars,
            "previous_keyframes": prev_keyframes,
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
        self._init_vector_store()
        self._meta = []
        for char in self.lattice.characters.values():
            for embedding in char.embeddings:
                self._add_to_memory(
                    embedding,
                    {
                        "type": "restored_character",
                        "character_id": char.id,
                        "name": char.name,
                    },
                )
        for shot in self.lattice.shots:
            for emb_data in shot.retrieved_context.get("keyframe_embeddings", []):
                embedding = Embedding.model_validate(emb_data)
                self._add_to_memory(
                    embedding,
                    {
                        "type": "restored_keyframe",
                        "shot_id": shot.id,
                        "shot_index": shot.index,
                    },
                )
        return self.lattice
