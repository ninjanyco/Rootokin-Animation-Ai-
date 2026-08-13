from pathlib import Path
import json
import tempfile
import unittest

from rootokin.animation_core.pipeline import run_pipeline
from rootokin.lattice.continuity_lattice import LatticeManager
from rootokin.lattice.nodes import CharacterNode
from rootokin.reference_engine.embedder import embed_images
from rootokin.three_d.loader import load_3d_model
from rootokin.three_d.stylizer import analyze_and_stylize_3d


class LatticeSmokeTests(unittest.TestCase):
    def test_memory_retrieval_returns_semantic_hits(self):
        manager = LatticeManager()
        character = CharacterNode(name="Pilot")
        manager.add_character(character)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "pilot.png"
            image_path.write_text("fake-image")
            embeddings = embed_images([image_path])
            manager.inject_image_refs(character.id, [image_path], embeddings)

            shot = manager.build_from_script([{"text": "Pilot enters", "characters": [character.id]}]).shots[0]
            retrieved = manager.retrieve_for_shot(shot)

        self.assertIn("semantic_memory", retrieved)
        self.assertGreaterEqual(len(retrieved["semantic_memory"]), 1)

    def test_pipeline_writes_lattice_and_memory_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            lattice_path = base / "lattice.json"
            script_path = base / "script.json"
            output_dir = base / "output"
            script_path.write_text(
                json.dumps(
                    [
                        {"text": "Beat one", "characters": []},
                        {"text": "Beat two", "characters": []},
                    ]
                )
            )

            run_pipeline(
                lattice_path=lattice_path,
                script_path=script_path,
                duration_minutes=0.2,
                output_dir=output_dir,
                backend="skyreels",
            )

            self.assertTrue(lattice_path.exists())
            loaded = LatticeManager().load(lattice_path)
            self.assertEqual(len(loaded.shots), 2)
            self.assertIn("shot_0000", loaded.memory_index)

    def test_3d_loader_and_stylizer_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            model = base / "hero.glb"
            model.write_text("fake-model")

            info = load_3d_model(model)
            self.assertEqual(info["format"], ".glb")

            result = analyze_and_stylize_3d(
                model_path=model,
                style_prompt="cinematic",
                style_ref_images=[],
                output_dir=base / "stylized",
            )

            self.assertIn("control_maps", result)
            self.assertIn("stylized_embeds", result)
            self.assertGreaterEqual(len(result["stylized_embeds"]), 1)


if __name__ == "__main__":
    unittest.main()
