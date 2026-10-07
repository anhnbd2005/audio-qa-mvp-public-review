# audio_qa_mvp

Deterministic audio-QA planning, language validation, generation, audit, and
release tooling. Gold answers come from dataset metadata; production QA
generation performs no external LLM calls and uses logical audio references.

## Current workflow

1. Profile the dataset.
2. Discover semantic question styles.
3. Prove semantic feasibility and capacity.
4. Validate language coverage and run Language Preflight.
5. Generate production QA from the accepted plan.
6. Run the final content audit.
7. Publish the release.

The canonical Vietnamese language system is under `resources/language/`.
There are no numbered runtime language bundles or compatibility fallbacks.

Run the permanent language checks:

```powershell
python -m src.language_preflight registry
python -m src.language_preflight dataset --dataset vimd
```

Run all tests without network, audio decoding, or model calls:

```powershell
pytest -q
```

The canonical ViMD package is `outputs/releases/vimd/final/`. Its `audio`
values are opaque logical references; physical audio is not part of the QA
package.

See `ARCHITECTURE.md` for the production contract.
