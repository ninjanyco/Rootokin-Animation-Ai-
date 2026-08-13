from pathlib import Path
from typing import List, Optional

import typer

from rootokin.animation_core.pipeline import run_pipeline
from rootokin.lattice.continuity_lattice import LatticeManager
from rootokin.lattice.nodes import CharacterNode, StyleNode, WorldNode
from rootokin.reference_engine.embedder import embed_images
from rootokin.three_d.stylizer import analyze_and_stylize_3d

app = typer.Typer(help="Rootokin Animation Engine – continuity-first open animation")


@app.command()
def init_lattice(
    output: Path = typer.Option(Path("lattice.json"), help="Where to save the lattice"),
    world_name: str = "rootokin_world",
    style_name: str = "cosmic_continuity",
    style_prompt: str = "cinematic, coherent, non-extractive, continuous world",
) -> None:
    """Create an empty continuity lattice."""
    manager = LatticeManager()
    manager.lattice.world = WorldNode(name=world_name)
    manager.lattice.style = StyleNode(name=style_name, prompt=style_prompt)
    manager.save(output)
    typer.echo(f"Lattice written to {output}")


@app.command()
def add_character(
    lattice_path: Path,
    name: str,
    image_dir: Optional[Path] = None,
    model_3d: Optional[Path] = None,
) -> None:
    """Add a character and optionally inject image and 3D references."""
    manager = LatticeManager()
    manager.load(lattice_path)
    character = CharacterNode(name=name)
    manager.add_character(character)

    if image_dir and image_dir.exists():
        images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))
        if images:
            embeddings = embed_images(images)
            manager.inject_image_refs(character.id, images, embeddings)
            typer.echo(f"Injected {len(images)} reference images")

    if model_3d and model_3d.exists():
        result = analyze_and_stylize_3d(
            model_path=model_3d,
            style_prompt=manager.lattice.style.prompt,
            style_ref_images=[],
            output_dir=Path("output/3d_stylized") / name,
        )
        manager.inject_3d(
            character.id,
            model_3d,
            result["control_maps"],
            result["stylized_embeds"],
        )
        typer.echo("Injected 3D model and stylized control maps")

    manager.save(lattice_path)
    typer.echo(f"Character '{name}' added. Lattice updated.")


@app.command()
def run(
    lattice_path: Path,
    script: Path,
    duration: float = 5.0,
    output_dir: Path = Path("output/run"),
    backend: str = "skyreels",
    known_characters: Optional[List[str]] = typer.Option(
        None,
        "--known-character",
        "--characters",
        help="Known character names to bias script parsing; repeat the option to pass multiple names.",
    ),
) -> None:
    """Build shots from script and run continuity-conditioned generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        lattice_path=lattice_path,
        script_path=script,
        duration_minutes=duration,
        output_dir=output_dir,
        backend=backend,
        known_characters=known_characters,
    )
    typer.echo(f"Pipeline finished → {output_dir}")


if __name__ == "__main__":
    app()
