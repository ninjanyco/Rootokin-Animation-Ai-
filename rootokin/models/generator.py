"""
Rootokin Generator – ensemble of open-source long-form video models.
Provides real loading stubs + lattice conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from rootokin.lattice.nodes import ContinuityLattice, ShotNode

if TYPE_CHECKING:
    from rootokin.lattice.continuity_lattice import LatticeManager


class RootokinGenerator:
    """
    Preferred backends (Aug 2026):
      1. skyreels   – best multi-reference + extension (SkyReels-V3)
      2. longcat    – native minutes-long continuation (LongCat-Video)
      3. ltx        – highest local quality + native audio (LTX-2.3)
      4. wan        – strong open character consistency (Wan 2.2)
    """

    def __init__(self, preferred_backend: str = "skyreels", device: Optional[str] = None):
        self.backend = preferred_backend.lower()
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.pipelines: Dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading stubs (call once)
    # ------------------------------------------------------------------
    def load(self, force: bool = False):
        if self._loaded and not force:
            return

        print(f"[generator] Loading backend: {self.backend} on {self.device}")

        if self.backend == "skyreels":
            self._load_skyreels()
        elif self.backend == "longcat":
            self._load_longcat()
        elif self.backend == "ltx":
            self._load_ltx()
        elif self.backend == "wan":
            self._load_wan()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._loaded = True

    def _load_skyreels(self):
        """
        SkyReels-V3 official pipelines.
        Requires: git clone https://github.com/SkyworkAI/SkyReels-V3
                  and models from HuggingFace (Skywork/SkyReels-V3-*)
        """
        try:
            from skyreels_v3.pipelines.reference_to_video_pipeline import ReferenceToVideoPipeline
            from skyreels_v3.pipelines.single_shot_extension_pipeline import SingleShotExtensionPipeline

            model_id = "Skywork/SkyReels-V3-Reference2Video"
            self.pipelines["r2v"] = ReferenceToVideoPipeline(
                model_path=model_id,
                offload=True,
                low_vram=True,
            )
            self.pipelines["extend"] = SingleShotExtensionPipeline(
                model_path="Skywork/SkyReels-V3-Video-Extension",
                offload=True,
                low_vram=True,
            )
            print("[generator] SkyReels-V3 pipelines ready")
        except Exception as exc:
            print(f"[generator] SkyReels load failed (install the official repo): {exc}")
            self.pipelines["skyreels_stub"] = None

    def _load_longcat(self):
        """
        LongCat-Video (meituan-longcat/LongCat-Video).
        Official loading pattern uses their LongCatVideoPipeline.
        """
        try:
            import importlib

            importlib.import_module("transformers")
            self.pipelines["longcat"] = None
            print("[generator] LongCat-Video stub registered (fill in official pipeline)")
        except Exception as exc:
            print(f"[generator] LongCat load failed: {exc}")

    def _load_ltx(self):
        """
        LTX-2.3 via Diffusers (cleanest open integration).
        """
        try:
            import diffusers

            pipeline_cls = getattr(diffusers, "LTXVideoPipeline", None) or getattr(diffusers, "LTXPipeline", None)
            if pipeline_cls is None:
                raise ImportError("No LTX pipeline class found in diffusers")

            pipe = pipeline_cls.from_pretrained(
                "Lightricks/LTX-2.3",
                torch_dtype=torch.bfloat16 if torch is not None else None,
            )
            pipe.enable_model_cpu_offload()
            self.pipelines["ltx"] = pipe
            print("[generator] LTX-2.3 Diffusers pipeline loaded")
        except Exception as exc:
            print(f"[generator] LTX load failed (pip install -U diffusers): {exc}")
            self.pipelines["ltx"] = None

    def _load_wan(self):
        """
        Wan 2.2 open weights via Diffusers.
        """
        try:
            from diffusers import AutoencoderKLWan, WanPipeline

            model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.bfloat16 if torch is not None else None,
            )
            pipe.to(self.device)
            self.pipelines["wan_t2v"] = pipe

            print("[generator] Wan 2.2 Diffusers pipelines loaded")
        except Exception as exc:
            print(f"[generator] Wan load failed: {exc}")
            self.pipelines["wan_t2v"] = None

    # ------------------------------------------------------------------
    # Lattice conditioning (unchanged logic, now feeds real models)
    # ------------------------------------------------------------------
    def condition_from_lattice(
        self,
        lattice: Union["LatticeManager", ContinuityLattice],
        shot: ShotNode,
    ) -> Dict[str, Any]:
        lattice_data = getattr(lattice, "lattice", lattice)
        if hasattr(lattice, "retrieve_for_shot"):
            ctx = lattice.retrieve_for_shot(shot)
        else:
            ctx = {
                "world": lattice_data.world,
                "style": lattice_data.style,
                "characters": {
                    cid: lattice_data.characters[cid] for cid in shot.characters if cid in lattice_data.characters
                },
                "previous_keyframes": [],
                "physics": lattice_data.world.physics_rules,
                "semantic_memory": [],
            }

        physics = ctx.get("physics", ctx.get("physics_constraints", {}))
        semantic_memory = ctx.get("semantic_memory", [])
        prompt = self._build_prompt(ctx, shot)

        ref_images: List[Path] = []
        control_maps: Dict[str, Path] = {}
        for char in ctx["characters"].values():
            ref_images.extend(char.reference_images)
            control_maps.update(char.control_maps)

        return {
            "prompt": prompt,
            "negative_prompt": "blurry, deformed, inconsistent character, flickering, low quality",
            "ref_images": ref_images[:4],
            "control_maps": control_maps,
            "style_prompt": ctx["style"].prompt,
            "previous_keyframes": ctx["previous_keyframes"],
            "physics": physics,
            "semantic_memory": semantic_memory,
        }

    def _build_prompt(self, ctx: Dict[str, Any], shot: ShotNode) -> str:
        chars = ", ".join(c.name for c in ctx["characters"].values()) or "main character"
        return (
            f"{shot.script_beat}. "
            f"Characters: {chars}. "
            f"Artistic style: {ctx['style'].prompt}. "
            "Maintain perfect visual continuity, same identity, lighting and world rules."
        )

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------
    def generate_shot(
        self,
        lattice: Union["LatticeManager", ContinuityLattice],
        shot: ShotNode,
        output_dir: Path,
        duration_sec: float = 5.0,
        seed: int = 42,
    ) -> Path:
        self.load()
        cond = self.condition_from_lattice(lattice, shot)
        shot.retrieved_context = cond
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"shot_{shot.index:04d}.mp4"

        print(f"[generator] Generating shot {shot.index} with {self.backend}")
        print(f"  Prompt: {cond['prompt'][:80]}...")

        try:
            if self.backend == "skyreels" and "r2v" in self.pipelines:
                video_path.touch()
                print("  → SkyReels call stubbed – fill with real pipe.generate_video(...)")
            elif self.backend == "ltx" and self.pipelines.get("ltx"):
                video_path.touch()
                print("  → LTX call stubbed – ready for real Diffusers call")
            elif self.backend == "wan" and self.pipelines.get("wan_t2v"):
                video_path.touch()
                print("  → Wan call stubbed")
            else:
                video_path.touch()
                print("  → Placeholder video created (model not fully loaded)")
        except Exception as exc:
            print(f"  Generation error: {exc}")
            video_path.touch()

        shot.generated_video_path = video_path
        return video_path
