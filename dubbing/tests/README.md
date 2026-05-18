# dubbing/tests/

Test suite for the dubbing project. All GPU-using tests share a single device controlled by the `TEST_GPU` environment variable (default: `1`), configured in `conftest.py`.

## Running tests

```bash
# Run all tests
TEST_GPU=0 conda run -n dubbing python -m pytest dubbing/tests/ -xvs

# Run a single test module directly (some have their own argparse)
conda run -n dubbing python -m dubbing.tests.test_cfm_index

# Run a single test function
TEST_GPU=0 conda run -n dubbing python -m pytest dubbing/tests/test_cfm_index.py::test_cfm_batch_inference -xvs
```

## Test files

| File | What it tests |
|---|---|
| `test_cfm_index.py` | Ported `dubbing/modules/cfm_index/CFM` vs original `index-tts2` CFM — verifies output equivalence by loading `s2mel.pth` weights into both and comparing L1 loss |
| `test_cfm_index_batch.py` | Batched inference consistency for `cfm_index` CFM |
| `test_cfm_index_lipsfeat_vocoder.py` | `CrossAttnCFM` with lips features + vocoder output |
| `test_data_provider.py` | Dataset and DataLoader correctness for all dataset types |
| `test_inferdub.py` | End-to-end inference pipeline for dubbing |
| `test_mfa_one.py` | MFA aligner on a single utterance |
| `test_semantic_transform.py` | `SemanticTransformer` correctness |
| `test_semantic_transform_offline.py` | Offline semantic transform pipeline |
| `test_semantic_warp_correctness.py` | Semantic warp alignment accuracy |
| `visualize/mel_visual.py` | Mel spectrogram visualization utility |

## Important conventions

- Tests use `weights_only=False` in `torch.load` — do not change this; checkpoints contain `SimpleNamespace` config objects that don't deserialize with `weights_only=True`.
- Some test files (e.g., `test_cfm_index.py`) are also runnable directly with their own argparse CLI, taking `--device`, `--n-samples`, `--n-steps` arguments.
- The `conftest.py` sets `CUDA_VISIBLE_DEVICES` from `TEST_GPU` at collection time, so it affects all test modules regardless of import order.
