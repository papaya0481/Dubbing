"""Pytest module: vocode x1_mel from cfm_index_phase1_for_lipsfeat samples.

This test loads Dataset_CFM_Index_Phase1_ForLipsFeat through the registered
`cfm_index_phase1_for_lipsfeat` data provider, randomly selects up to 50
samples, converts each `x1_mel` to waveform with the same BigVGAN setup used
in `dubbing/tests/test_cfm_index.py`, and saves:

- <output_dir>/wav/<stem>.wav
- <output_dir>/textgrid/<stem>.TextGrid (copied from lips_textgrid)

The test is skipped automatically when required external paths are unavailable.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchaudio

OmegaConf = pytest.importorskip("omegaconf").OmegaConf


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DUB_ROOT = _HERE.parent
_PROJ_ROOT = _DUB_ROOT.parent
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("HF_HUB_CACHE", str(_PROJ_ROOT / "checkpoints" / "hf_cache"))


# ---------------------------------------------------------------------------
# External resources
# ---------------------------------------------------------------------------
_FLOW_DATASET_ROOT = Path("/data2/ruixin/datasets/flow_dataset/MELD")
_MODEL_DIR = Path("/data2/ruixin/index-tts2/checkpoints")
_CONFIG_PATH = _DUB_ROOT / "configs" / "default_cfm_index.yaml"

_SKIP = pytest.mark.skipif(
    not _FLOW_DATASET_ROOT.exists() or not _MODEL_DIR.exists() or not _CONFIG_PATH.exists(),
    reason=(
        "Required data/model/config path missing. "
        f"flow_dataset={_FLOW_DATASET_ROOT}, model_dir={_MODEL_DIR}, cfg={_CONFIG_PATH}"
    ),
)


def _make_args() -> SimpleNamespace:
    from config import load_config

    args = load_config(str(_CONFIG_PATH))
    args.data.dataset = "cfm_index_phase1_for_lipsfeat"
    args.data.flow_dataset_path = str(_FLOW_DATASET_ROOT)
    args.data.tier_name = "phones"
    args.data.batch_size = 1
    args.data.num_workers = 0
    args.data.max_samples = 50
    args.system.seed = getattr(args.system, "seed", 2026)
    return args


def _output_root() -> Path:
    return Path("/data2/ruixin/ours/test_outputs/cfm_index_lipsfeat_vocoder")


@_SKIP
def test_cfm_index_phase1_for_lipsfeat_vocode_50_samples() -> None:
    """Load dataset, randomly pick 50 samples, save wav + lips TextGrid."""
    from data_provider.data_factory import data_provider
    from indextts.s2mel.modules.bigvgan import bigvgan

    args = _make_args()
    try:
        dataset, _ = data_provider(args, "train")
    except RuntimeError as e:
        if "No valid intersected samples" in str(e):
            pytest.skip(str(e))
        raise

    assert len(dataset) > 0, "cfm_index_phase1_for_lipsfeat train split returned 0 samples"

    n_pick = min(50, len(dataset))
    rng = random.Random(2026)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:n_pick]

    cfg = OmegaConf.load(_MODEL_DIR / "config.yaml")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vocoder = bigvgan.BigVGAN.from_pretrained(cfg.vocoder.name, use_cuda_kernel=False)
    vocoder = vocoder.to(device)
    vocoder.remove_weight_norm()
    vocoder.eval()

    out_root = _output_root()
    out_wav_dir = out_root / "wav"
    out_tg_dir = out_root / "textgrid"
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_tg_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in indices:
        item = dataset[idx]
        stem = str(item["stem"])

        x1_mel = item["x1_mel"]
        if x1_mel.dim() != 2:
            raise AssertionError(f"x1_mel for {stem} must be 2-D [80, T], got shape={tuple(x1_mel.shape)}")

        with torch.no_grad():
            wav = vocoder(x1_mel.unsqueeze(0).to(device).float()).squeeze(1)

        wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu().to(torch.int16)
        wav_path = out_wav_dir / f"{stem}.wav"
        torchaudio.save(str(wav_path), wav, 22050)

        lips_tg = Path(item["lips_textgrid"])
        assert lips_tg.exists(), f"lips_textgrid not found for {stem}: {lips_tg}"
        tg_dst = out_tg_dir / f"{stem}{lips_tg.suffix}"
        shutil.copy2(lips_tg, tg_dst)

        out_wav_src = Path(item["out_wav"])
        if out_wav_src.exists():
            shutil.copy2(out_wav_src, out_wav_dir / f"{stem}_origin.wav")

        saved += 1

    assert saved == n_pick, f"Saved {saved} samples, expected {n_pick}"
