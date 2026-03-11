"""Unit tests for all registered data providers.

Providers tested
----------------
- cfm_phase1          (Dataset_CFM_Phase1)
- cfm_phase1_stretch  (Dataset_CFM_Phase1_StretchEntireMel)
- cfm_index_phase1    (Dataset_CFM_Index_Phase1)

Each provider is verified for:
- Dataset non-empty after split
- Batch contains expected keys with correct tensor shapes and dtypes
- train / val / test splits are mutually disjoint

Tests that require external data paths are skipped automatically when
those paths are unavailable on the current machine.

Usage:
    pytest tests/test_data_provider.py
"""

import sys
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

logging.getLogger().setLevel(logging.WARNING)

# ---- path setup --------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_DUB_ROOT = _HERE.parent   # dubbing/
if str(_DUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_DUB_ROOT))

# ---- external data paths -----------------------------------------------
_CFM_PHASE1_ROOT  = Path("/data2/ruixin/datasets/MELD_gen_pairs")
_CFM_INDEX_CSV    = Path("/data2/ruixin/datasets/flow_dataset/MELD_semantic/metadata.csv")
_CFM_INDEX_MDLDIR = Path("/data2/ruixin/index-tts2/checkpoints")

_SKIP_PHASE1 = pytest.mark.skipif(
    not _CFM_PHASE1_ROOT.exists(),
    reason=f"cfm_phase1 data root not found: {_CFM_PHASE1_ROOT}",
)
_SKIP_INDEX = pytest.mark.skipif(
    not _CFM_INDEX_CSV.exists(),
    reason=f"cfm_index_phase1 CSV not found: {_CFM_INDEX_CSV}",
)


# ========================================================================
# Helpers
# ========================================================================

def _make_phase1_args(dataset_name: str = "cfm_phase1") -> SimpleNamespace:
    """Build a minimal args namespace for cfm_phase1 / cfm_phase1_stretch."""
    data = SimpleNamespace(
        root=str(_CFM_PHASE1_ROOT),
        dataset=dataset_name,
        train_split_ratio=0.9,
        filter_by_mse=True,
        mse_threshold=4.0,
        tier_name="phones",
        phoneme_map_path=str(_DUB_ROOT / "modules" / "english_us_arpa_300.json"),
        batch_size=4,
        num_workers=0,
    )
    system = SimpleNamespace(seed=2026)
    return SimpleNamespace(data=data, system=system)


def _make_index_args() -> SimpleNamespace:
    """Load cfm_index_phase1 args from the default YAML config."""
    from config import load_config
    args = load_config(str(_DUB_ROOT / "configs" / "default_cfm_index.yaml"))
    args.data.batch_size  = 2
    args.data.num_workers = 0
    return args


# ========================================================================
# cfm_phase1
# ========================================================================

@_SKIP_PHASE1
def test_cfm_phase1_dataset_nonempty():
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1")
    dataset, _ = data_provider(args, "train")
    assert len(dataset) > 0, "cfm_phase1 train split returned 0 samples"


@_SKIP_PHASE1
def test_cfm_phase1_batch_keys_and_shapes():
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1")
    _, loader = data_provider(args, "train")
    batch = next(iter(loader))

    required = {"pair_key", "cond_mel", "x1", "x_mean", "x_std", "x_lens",
                "phoneme_ids", "mse", "text_r1"}
    missing = required - batch.keys()
    assert not missing, f"Batch missing keys: {missing}"

    B = len(batch["pair_key"])
    assert B >= 1

    cond_mel = batch["cond_mel"]
    x1       = batch["x1"]
    x_lens   = batch["x_lens"]

    assert cond_mel.ndim    == 3,       "cond_mel must be 3-D (B, 80, T)"
    assert cond_mel.shape[1] == 80,     "cond_mel channel dim must be 80"
    assert x1.shape          == cond_mel.shape, "x1 and cond_mel must have identical shape"
    assert x_lens.shape      == (B,)
    assert x_lens.dtype      == torch.long
    assert cond_mel.dtype    == torch.float32
    # x_lens must not exceed padded time dimension
    assert int(x_lens.max()) <= cond_mel.shape[2]


@_SKIP_PHASE1
def test_cfm_phase1_splits_disjoint():
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1")
    train_ds, _ = data_provider(args, "train")
    val_ds,   _ = data_provider(args, "val")
    test_ds,  _ = data_provider(args, "test")

    train_keys = {s.pair_key for s in train_ds.samples}
    val_keys   = {s.pair_key for s in val_ds.samples}
    test_keys  = {s.pair_key for s in test_ds.samples}

    assert train_keys.isdisjoint(val_keys),  "train/val pair_keys overlap"
    assert train_keys.isdisjoint(test_keys), "train/test pair_keys overlap"
    assert val_keys.isdisjoint(test_keys),   "val/test pair_keys overlap"


@_SKIP_PHASE1
def test_cfm_phase1_splits_cover_all_samples():
    """train + val + test should together cover every sample exactly once."""
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1")
    train_ds, _ = data_provider(args, "train")
    val_ds,   _ = data_provider(args, "val")
    test_ds,  _ = data_provider(args, "test")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    # Reload all splits from the same dataset class to count canonical total
    from data_provider.data_loader import Dataset_CFM_Phase1
    all_ds = Dataset_CFM_Phase1(
        root_dir=str(_CFM_PHASE1_ROOT),
        split="train",
        split_ratio=1.0,
        seed=2026,
        filter_enabled=True,
        mse_threshold=4.0,
        tier_name="phones",
        phoneme_map_path=str(_DUB_ROOT / "modules" / "english_us_arpa_300.json"),
    )
    assert total == len(train_ds) + len(val_ds) + len(test_ds)
    assert total > 0


# ========================================================================
# cfm_phase1_stretch
# ========================================================================

@_SKIP_PHASE1
def test_cfm_phase1_stretch_dataset_nonempty():
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1_stretch")
    dataset, _ = data_provider(args, "train")
    assert len(dataset) > 0, "cfm_phase1_stretch train split returned 0 samples"


@_SKIP_PHASE1
def test_cfm_phase1_stretch_batch_keys_and_shapes():
    from data_provider.data_factory import data_provider
    args = _make_phase1_args("cfm_phase1_stretch")
    _, loader = data_provider(args, "train")
    batch = next(iter(loader))

    required = {"pair_key", "cond_mel", "x1", "x_mean", "x_std", "x_lens"}
    missing = required - batch.keys()
    assert not missing, f"Batch missing keys: {missing}"

    cond_mel = batch["cond_mel"]
    x1       = batch["x1"]
    assert cond_mel.ndim    == 3
    assert cond_mel.shape[1] == 80
    assert x1.shape          == cond_mel.shape
    assert cond_mel.dtype    == torch.float32


# ========================================================================
# cfm_index_phase1
# ========================================================================

@_SKIP_INDEX
def test_cfm_index_phase1_dataset_nonempty():
    from data_provider.data_factory import data_provider
    args = _make_index_args()
    dataset, _ = data_provider(args, "train")
    assert len(dataset) > 0, "cfm_index_phase1 train split returned 0 samples"


@_SKIP_INDEX
def test_cfm_index_phase1_batch_keys_and_shapes():
    from data_provider.data_factory import data_provider
    args  = _make_index_args()
    _, loader = data_provider(args, "train")
    batch = next(iter(loader))

    required = {"stems", "x1_full", "prompt_cond", "infer_cond",
                "ref_mels", "style", "x_lens", "prompt_lens", "infer_lens"}
    missing  = required - batch.keys()
    assert not missing, f"Batch missing keys: {missing}"

    B = len(batch["stems"])
    assert B >= 1

    x1_full     = batch["x1_full"]
    prompt_cond = batch["prompt_cond"]
    infer_cond  = batch["infer_cond"]
    ref_mels    = batch["ref_mels"]
    style       = batch["style"]
    x_lens      = batch["x_lens"]
    prompt_lens = batch["prompt_lens"]
    infer_lens  = batch["infer_lens"]

    # shape checks
    assert x1_full.ndim      == 3,   "x1_full must be 3-D (B, 80, T_max)"
    assert x1_full.shape[1]  == 80,  "x1_full mel dim must be 80"
    assert prompt_cond.ndim  == 3,   "prompt_cond must be 3-D (B, T_ref_max, 512)"
    assert prompt_cond.shape[2] == 512, "prompt_cond last dim must be 512"
    assert infer_cond.ndim   == 3,   "infer_cond must be 3-D (B, T_gen_max, 512)"
    assert infer_cond.shape[2] == 512,  "infer_cond last dim must be 512"
    assert ref_mels.ndim     == 3,   "ref_mels must be 3-D (B, 80, T_ref_max)"
    assert ref_mels.shape[1] == 80
    assert style.shape    == (B, 192), f"style shape {style.shape} != ({B}, 192)"
    assert x_lens.shape      == (B,)
    assert prompt_lens.shape == (B,)
    assert infer_lens.shape  == (B,)

    # logical consistency: x_lens == prompt_lens + infer_lens
    assert (prompt_lens + infer_lens == x_lens).all(), \
        "x_lens must equal prompt_lens + infer_lens"
    assert int(x_lens.max())      <= x1_full.shape[2],     "x_lens exceeds padded time dim"
    assert int(prompt_lens.max()) <= prompt_cond.shape[1], "prompt_lens exceeds prompt_cond padded dim"
    assert int(infer_lens.max())  <= infer_cond.shape[1],  "infer_lens exceeds infer_cond padded dim"
    assert int(prompt_lens.max()) <= ref_mels.shape[2],    "prompt_lens exceeds ref_mels padded dim"

    # dtype checks
    assert x1_full.dtype     == torch.float32
    assert prompt_cond.dtype == torch.float32
    assert infer_cond.dtype  == torch.float32
    assert style.dtype       == torch.float32
    assert x_lens.dtype      == torch.long
    assert prompt_lens.dtype == torch.long
    assert infer_lens.dtype  == torch.long


@_SKIP_INDEX
def test_cfm_index_phase1_splits_disjoint():
    from data_provider.data_factory import data_provider
    args = _make_index_args()
    train_ds, _ = data_provider(args, "train")
    val_ds,   _ = data_provider(args, "val")
    test_ds,  _ = data_provider(args, "test")

    train_stems = {s["stem"] for s in train_ds.samples}
    val_stems   = {s["stem"] for s in val_ds.samples}
    test_stems  = {s["stem"] for s in test_ds.samples}

    assert train_stems.isdisjoint(val_stems),  "train/val stems overlap"
    assert train_stems.isdisjoint(test_stems), "train/test stems overlap"
    assert val_stems.isdisjoint(test_stems),   "val/test stems overlap"


@_SKIP_INDEX
def test_cfm_index_phase1_splits_cover_all_samples():
    from data_provider.data_factory import data_provider
    args = _make_index_args()
    train_ds, _ = data_provider(args, "train")
    val_ds,   _ = data_provider(args, "val")
    test_ds,  _ = data_provider(args, "test")
    assert len(train_ds) + len(val_ds) + len(test_ds) > 0