from pathlib import Path
import tempfile
import unittest

from rootokin.generation.generator import RootokinGenerator
from rootokin.lattice.manager import LatticeManager
from rootokin.lattice.nodes import CharacterNode, Embedding


class LatticeTests(unittest.TestCase):
    def test_build_from_script_links_shots(self):
        manager = LatticeManager()
        lattice = manager.build_from_script(
            [
                {"text": "Hero enters", "characters": ["c1"]},
                {"text": "Villain appears", "characters": ["c2"]},
            ],
            duration_minutes=1,
        )

        self.assertEqual(len(lattice.shots), 2)
        self.assertIsNone(lattice.shots[0].previous_shot_id)
        self.assertEqual(lattice.shots[1].previous_shot_id, lattice.shots[0].id)

    def test_retrieve_for_shot_uses_previous_keyframes(self):
        manager = LatticeManager()
        lattice = manager.build_from_script(
            [{"text": "Beat1", "characters": []}, {"text": "Beat2", "characters": []}],
            duration_minutes=1,
        )
        lattice.shots[0].keyframe_paths = [Path("/tmp/kf1.png")]

        context = lattice.retrieve_for_shot(lattice.shots[1])
        self.assertEqual(context["previous_keyframes"], [Path("/tmp/kf1.png")])

    def test_save_and_load_roundtrip(self):
        manager = LatticeManager()
        character = CharacterNode(name="Ari")
        manager.lattice.add_character(character)
        manager.lattice.inject_image_refs(
            character.id,
            [Path("ref.png")],
            [Embedding(vector=[0.1, 0.2], source="image")],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "lattice.json"
            manager.save(path)
            loaded = manager.load(path)

        self.assertIn(character.id, loaded.characters)
        self.assertEqual(loaded.characters[character.id].reference_images, [Path("ref.png")])

    def test_generator_condition_collects_references_and_controls(self):
        manager = LatticeManager()
        character = CharacterNode(name="Ari")
        manager.lattice.add_character(character)

        manager.lattice.inject_image_refs(
            character.id,
            [Path("char.png")],
            [Embedding(vector=[0.2], source="image")],
        )
        manager.lattice.inject_3d(
            character.id,
            model_path=Path("char.glb"),
            control_maps={"depth": Path("depth.exr")},
            stylized_embeds=[Embedding(vector=[0.3], source="3d_render")],
        )

        shot = manager.build_from_script([{"text": "Ari runs", "characters": [character.id]}]).shots[0]
        generator = RootokinGenerator()

        condition = generator.condition_from_lattice(manager.lattice, shot)
        self.assertEqual(condition["ref_images"], [Path("char.png")])
        self.assertEqual(condition["control_maps"], {"depth": Path("depth.exr")})


if __name__ == "__main__":
    unittest.main()
