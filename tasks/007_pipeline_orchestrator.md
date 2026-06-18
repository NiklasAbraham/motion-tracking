# Task 007 – Pipeline Orchestrator

## Summary

Build the top-level orchestration module that chains all stages together and provides a single CLI entry point for the full video-to-3D-viewer workflow.

## Spec Reference

`specs/006_pipeline_orchestration.md`

## Deliverables

- `src/pipeline.py` – orchestration logic
- `tests/test_pipeline.py` – integration tests (with mocked stages)

## Acceptance Criteria

1. Single CLI command: `python -m src.pipeline <video.mp4> --output <dir>`.
2. Runs all stages in order: validate → 2D extract → map → lift → post-process → visualize.
3. Stores intermediate artifacts in working directory.
4. Supports resuming from last completed stage.
5. Reports progress per stage.
6. Passes configuration through to each stage.

## Dependencies (External)

- `argparse`, `pathlib`, `dataclasses` (stdlib)

## Dependencies (Internal)

- All other task modules (at runtime only — tested with mocks).

## Test Cases

| ID | Description | Input | Expected |
|----|-------------|-------|----------|
| T7.1 | Full pipeline mock | Mock video + mocked stages | All stages called in order |
| T7.2 | Resume from stage 3 | Working dir with stage 1-2 artifacts | Skips stages 1-2, starts at 3 |
| T7.3 | Invalid video | Non-existent file | Fails at validation, reports error |
| T7.4 | Missing checkpoint | Valid video, no model files | Fails at lifting, reports error |
| T7.5 | Output directory creation | Non-existent output dir | Creates directory automatically |
| T7.6 | Progress reporting | Full run | Progress callback called per stage |
| T7.7 | Custom config | Override filter params | Params passed through to post-processor |

## Parallelism

✅ Can be developed and tested independently using mocked stage functions. Integration testing requires other modules but unit tests are self-contained.
