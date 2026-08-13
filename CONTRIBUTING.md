# Contributing to Rootokin

## Philosophy
- Continuity and world coherence are non-negotiable.
- Prefer open-weight models and permissive licenses.
- Keep the lattice as the single source of truth.

## Development Setup

```bash
git clone https://github.com/ninjanyco/Rootokin-Animation-Ai-.git
cd Rootokin-Animation-Ai-
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install open_clip_torch faiss-cpu opencv-python-headless pillow torch
python -m unittest discover -s tests -v
```

## Project Structure
- `rootokin/lattice/` – continuity lattice data model and memory manager
- `rootokin/script_engine/` – script parsing into structured beats
- `rootokin/reference_engine/` – embedding generation for images and text
- `rootokin/three_d/` – 3D loading and stylization helpers
- `rootokin/models/` – backend generation adapters
- `rootokin/animation_core/` – pipeline and keyframe update loop
- `tests/` – regression and smoke coverage

## Coding Standards
- Make continuity-related changes in the lattice layer before adding backend-specific logic.
- Preserve graceful fallbacks when optional dependencies are unavailable.
- Prefer small, composable additions over tightly coupled pipeline changes.
- Keep prompts, metadata, and retrieved context structured and serializable.

## Testing
- Add or update targeted tests for any behavior change.
- Run `python -m unittest discover -s tests -v` before opening a pull request.
- If you add an optional integration, keep the default test path working without that dependency installed.

## Documentation and Examples
- Update `README.md` when the public workflow changes.
- Keep `examples/simple_run/story.txt` usable as a continuity demo.
- Document new vector stores, generation backends, or CLI options in the relevant guide.

## Pull Requests
When opening a pull request:

1. Explain the continuity impact.
2. Summarize any lattice, retrieval, or backend changes.
3. List the tests you ran.
4. Note any optional dependencies reviewers may need.

## Contribution Priorities
- Better long-form continuity retrieval
- Open-model backend integrations
- More reliable keyframe memory updates
- Stronger tests around persistence and temporal coherence
