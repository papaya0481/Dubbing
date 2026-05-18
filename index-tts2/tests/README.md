# index-tts2/tests/

Test suite for the IndexTTS2 reference implementation.

## Files

| File | Purpose |
|---|---|
| `regression_test.py` | Regression tests comparing generated output against reference results |
| `padding_test.py` | Tests for sequence padding behavior in the GPT model |
| `cases.jsonl` | Test cases: text/audio pairs for regression testing |

## Usage

```bash
cd index-tts2
PYTHONPATH=. uv run python tests/regression_test.py
PYTHONPATH=. uv run python tests/padding_test.py
```

These tests assume the IndexTTS checkpoints and any referenced sample audio files are already available under `checkpoints/` and `tests/`.
