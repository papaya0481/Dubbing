"""
test_semantic_transform_offline.py
===================================
Pytest-runnable test for semantic transform with 4 combinations of warp_type and grid_sample_mode.

Tests 4 situations:
1. warp_type=semantic, grid_sample_mode=nearest
2. warp_type=semantic, grid_sample_mode=bilinear
3. warp_type=cond, grid_sample_mode=nearest
4. warp_type=cond, grid_sample_mode=bilinear

Plus the original (no transform) for comparison.

Outputs:
- 5 wav files (4 transformed + 1 original) per sample
- 1 visualization with 5 mel spectrograms and aligned phoneme bars per sample

Usage:
    pytest dubbing/tests/test_semantic_transform_offline.py -v -s
"""

import sys
import os
import random
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torchaudio
import tgt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "index-tts2"))
sys.path.insert(0, str(project_root / "modules"))

from modules.mel_strech.meldataset import get_mel_spectrogram
from modules.semantic_stretch.semantic_transform import SemanticTransformer
from lips.data.phoneme_vocab import PhonemeVocab
from omegaconf import OmegaConf
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from logger import get_logger

logger = get_logger("test_semantic_transform_offline")


# =============================================================================
# Configuration
# =============================================================================

# Paths - adjust these to your actual data
FLOW_DATASET_PATH = Path("/data2/ruixin/ours/MELD")
SEMANTIC_ROOT = FLOW_DATASET_PATH / "semantic"
PREDICT_ROOT = FLOW_DATASET_PATH / "predict_results"
SOURCE_TG_DIR = SEMANTIC_ROOT / "audios" / "aligned"
LIPS_TG_DIR = PREDICT_ROOT / "textgrids"
DATA_ROOT = SEMANTIC_ROOT

# Model paths
MODEL_DIR = project_root / "index-tts2" / "ckpts" / "maskgct-s2a-base"
W2V_STAT = "w2v-bert-2.0.stat.pt"
S2MEL_CHECKPOINT = "s2mel_epoch_00100.pth"

# Number of random samples to test (can be overridden via environment variable)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))

# Output directory
OUTPUT_DIR = Path("/data2/ruixin/ours/test_outputs/semantic_transform_offline")

# Test configurations
TEST_CONFIGS = [
    ("semantic", "nearest"),
    ("semantic", "bilinear"),
    ("cond", "nearest"),
    ("cond", "bilinear"),
]

TIER_NAME = "phones"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEL_FPS = 22050.0 / 256.0  # 86.1328125


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def setup_output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="module")
def models():
    """Load all models once for all tests."""
    logger.info(f"Device: {DEVICE}")
    logger.info("Loading models...")

    cfg = OmegaConf.load(str(MODEL_DIR / "config.yaml"))

    # Load mel config
    from modules.mel_strech.meldataset import mel_spectrogram
    mel_h = mel_spectrogram.AttrDict({
        'sampling_rate': 22050,
        'n_fft': 1024,
        'num_mels': 80,
        'hop_size': 256,
        'win_size': 1024,
        'fmin': 0,
        'fmax': 8000,
    })

    # Load w2v-bert
    logger.info("Loading w2v-bert...")
    sem_model, sem_mean, sem_std = build_semantic_model(str(MODEL_DIR / W2V_STAT))
    sem_model = sem_model.to(DEVICE).eval()
    sem_mean = sem_mean.to(DEVICE)
    sem_std = sem_std.to(DEVICE)

    # Load semantic codec
    logger.info("Loading semantic codec...")
    from huggingface_hub import hf_hub_download
    import safetensors
    codec = build_semantic_codec(cfg.semantic_codec)
    codec_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(codec, codec_ckpt)
    sem_codec = codec.to(DEVICE).eval()

    # Load length regulator and CFM
    logger.info("Loading s2mel model...")
    s2mel_model = MyModel(cfg.s2mel, use_gpt_latent=False)
    s2mel_model, _, _, _ = load_checkpoint2(
        s2mel_model, None, str(MODEL_DIR / S2MEL_CHECKPOINT),
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    len_reg = s2mel_model.models["length_regulator"].to(DEVICE).eval()
    cfm = s2mel_model.models["cfm"].to(DEVICE).eval()

    # Load phoneme vocab
    vocab = PhonemeVocab()

    logger.info("Models loaded successfully")

    return {
        'mel_h': mel_h,
        'len_reg': len_reg,
        'cfm': cfm,
        'vocab': vocab,
    }


# =============================================================================
# Helper functions
# =============================================================================

def discover_available_samples(seed: int = 42) -> List[str]:
    """Discover all available samples that have all required files.

    Returns a randomly sampled list of stems.
    """
    # Find all .pt files in the semantic audios directory
    pt_files = list((DATA_ROOT / "audios").glob("*.pt"))

    available_stems = []
    for pt_file in pt_files:
        stem = pt_file.stem

        # Check if all required files exist
        out_wav = DATA_ROOT / "audios" / f"{stem}.wav"

        source_tg = None
        for ext in (".TextGrid", ".textgrid"):
            p = SOURCE_TG_DIR / f"{stem}{ext}"
            if p.exists():
                source_tg = p
                break

        lips_tg = None
        for ext in (".TextGrid", ".textgrid"):
            p = LIPS_TG_DIR / f"{stem}{ext}"
            if p.exists():
                lips_tg = p
                break

        # Only include if all files exist
        if out_wav.exists() and source_tg is not None and lips_tg is not None:
            available_stems.append(stem)

    logger.info(f"Found {len(available_stems)} available samples")

    # Randomly sample
    random.seed(seed)
    num_to_sample = min(NUM_SAMPLES, len(available_stems))
    sampled = random.sample(available_stems, num_to_sample)

    logger.info(f"Randomly sampled {len(sampled)} samples for testing")
    return sampled


def load_sample_data(stem: str) -> dict:
    """Load all necessary data for a sample."""
    # Find files
    out_pt = DATA_ROOT / "audios" / f"{stem}.pt"
    out_wav = DATA_ROOT / "audios" / f"{stem}.wav"

    source_tg = None
    lips_tg = None

    for ext in (".TextGrid", ".textgrid"):
        p = SOURCE_TG_DIR / f"{stem}{ext}"
        if p.exists():
            source_tg = p
            break

    for ext in (".TextGrid", ".textgrid"):
        p = LIPS_TG_DIR / f"{stem}{ext}"
        if p.exists():
            lips_tg = p
            break

    if not out_pt.exists():
        raise FileNotFoundError(f"Semantic codes not found: {out_pt}")
    if not out_wav.exists():
        raise FileNotFoundError(f"Audio not found: {out_wav}")
    if source_tg is None:
        raise FileNotFoundError(f"Source TextGrid not found for {stem}")
    if lips_tg is None:
        raise FileNotFoundError(f"Lips TextGrid not found for {stem}")

    return {
        "stem": stem,
        "out_pt": str(out_pt),
        "out_wav": str(out_wav),
        "source_textgrid": str(source_tg),
        "lips_textgrid": str(lips_tg),
    }


def normalize_textgrid_phonemes(tg: tgt.TextGrid, tier_name: str, vocab: PhonemeVocab):
    """Map phonemes to VFA domain."""
    try:
        tier = tg.get_tier_by_name(tier_name)
    except Exception:
        return

    for iv in tier:
        txt = (iv.text or "").strip()
        if txt == "":
            continue
        vfa_id = vocab.arpabet_to_vfa_id(txt)
        iv.text = vocab.vfa_id_to_arpabet(vfa_id)


def textgrid_to_frame_phonemes(tg: tgt.TextGrid, tier_name: str, vocab: PhonemeVocab,
                                total_frames: int, fps: float = MEL_FPS) -> Tuple[np.ndarray, List[str]]:
    """Convert TextGrid to frame-level phoneme IDs and text labels.

    Args:
        tg: TextGrid object
        tier_name: Name of the tier to use
        vocab: PhonemeVocab for mapping
        total_frames: Total number of mel frames
        fps: Mel frame rate (22050/256 = 86.1328125)

    Returns:
        phoneme_ids: (total_frames,) array of phoneme IDs
        phoneme_texts: List of phoneme text labels
    """
    tier = tg.get_tier_by_name(tier_name)
    phoneme_ids = np.ones(total_frames, dtype=np.int32)  # Default to SIL

    for iv in tier:
        txt = (iv.text or "").strip()
        if txt in ("", "sp", "sil", "<eps>"):
            ph_id = 1  # SIL
        else:
            ph_id = vocab.arpabet_to_vfa_id(txt)

        # Convert time to frame indices
        start_frame = int(iv.start_time * fps)
        end_frame = int(iv.end_time * fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))

        if end_frame > start_frame:
            phoneme_ids[start_frame:end_frame] = ph_id

    # Convert IDs to text
    phoneme_texts = []
    for pid in phoneme_ids:
        try:
            phoneme_texts.append(vocab.vfa_id_to_arpabet(int(pid)))
        except:
            phoneme_texts.append("<unk>")

    return phoneme_ids, phoneme_texts


def save_multi_mel_visualization(
    mel_list: List[torch.Tensor],
    phoneme_ids_list: List[np.ndarray],
    phoneme_texts_list: List[List[str]],
    labels: List[str],
    image_path: str,
    title: str = "",
):
    """Plot 5 mel spectrograms with phoneme alignment bars."""
    n_plots = len(mel_list)
    assert n_plots == len(phoneme_ids_list) == len(labels)

    fig_height = 3 * n_plots
    fig = plt.figure(figsize=(20, fig_height))

    # Create grid: each row has [mel, phoneme bar]
    gs = GridSpec(n_plots * 2, 2, figure=fig,
                  height_ratios=[3, 1] * n_plots,
                  width_ratios=[0.97, 0.03],
                  hspace=0.4, wspace=0.05)

    for idx in range(n_plots):
        mel = mel_list[idx].detach().cpu().squeeze(0).numpy()
        ph_ids = phoneme_ids_list[idx]
        frame_phonemes = phoneme_texts_list[idx]
        label = labels[idx]

        total_frames = mel.shape[-1]

        # Create subplots
        row_base = idx * 2
        ax_mel = fig.add_subplot(gs[row_base, 0])
        ax_ph = fig.add_subplot(gs[row_base + 1, 0], sharex=ax_mel)

        # Only add colorbar for first plot
        if idx == 0:
            ax_cbar = fig.add_subplot(gs[row_base, 1])

        # Plot mel spectrogram
        mel_img = ax_mel.imshow(
            mel, origin="lower", aspect="auto",
            interpolation="nearest", cmap="magma",
            extent=[-0.5, total_frames - 0.5, -0.5, mel.shape[0] - 0.5],
        )
        ax_mel.set_title(f"{label}")
        ax_mel.set_ylabel("Mel Bins")

        if idx == 0:
            fig.colorbar(mel_img, cax=ax_cbar).set_label("Log-Mel")

        # Plot phoneme bar
        ax_ph.imshow(
            np.expand_dims(ph_ids, axis=0),
            aspect="auto", interpolation="nearest", cmap="tab20",
            extent=[-0.5, total_frames - 0.5, 0.0, 1.0],
        )
        ax_ph.set_yticks([])
        ax_ph.set_ylabel("Phoneme")
        if idx == n_plots - 1:
            ax_ph.set_xlabel("Frame Index")

        # Set limits
        x_left, x_right = -0.5, total_frames - 0.5
        ax_mel.set_xlim(x_left, x_right)
        ax_ph.set_xlim(x_left, x_right)

        # Add phoneme text labels
        segments, seg_start = [], 0
        for frame_idx in range(1, total_frames + 1):
            if frame_idx == total_frames or frame_phonemes[frame_idx] != frame_phonemes[seg_start]:
                segments.append((seg_start, frame_idx - 1, frame_phonemes[seg_start]))
                seg_start = frame_idx

        label_fontsize = 8 if total_frames > 180 else 10
        for start, end, ph in segments:
            ax_ph.text(
                (start + end) / 2.0, 0.5, str(ph),
                ha="center", va="center", fontsize=label_fontsize, color="black",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.6},
            )

        # X-axis ticks
        tick_step = max(1, total_frames // 20)
        xticks = np.arange(0, total_frames, tick_step)
        ax_ph.set_xticks(xticks)
        ax_ph.set_xticklabels([str(i) for i in xticks], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()
    plt.savefig(image_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visualization: {image_path}")


def mel_to_wav_placeholder(mel: torch.Tensor, output_path: Path):
    """Placeholder for mel-to-wav conversion. Saves a silent wav file."""
    # In a real implementation, you would use a vocoder here
    # For now, just create a silent wav file as placeholder
    duration_sec = mel.shape[-1] * 256 / 22050
    samples = int(duration_sec * 22050)
    wav = torch.zeros(1, samples)
    torchaudio.save(str(output_path), wav, 22050)
    logger.info(f"Saved placeholder wav: {output_path}")


# =============================================================================
# Test functions
# =============================================================================

def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on available samples."""
    if "sample_stem" in metafunc.fixturenames:
        # Discover available samples
        available_samples = discover_available_samples()
        metafunc.parametrize("sample_stem", available_samples)


def test_semantic_transform_sample(sample_stem, models, setup_output_dir):
    """Test semantic transform for a single sample with all 4 configurations + original."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing sample: {sample_stem}")
    logger.info(f"{'='*60}")

    # Load sample data
    try:
        sample_data = load_sample_data(sample_stem)
    except FileNotFoundError as e:
        pytest.skip(f"Sample data not found: {e}")

    # Load semantic codes
    s_infer = torch.load(sample_data["out_pt"], map_location="cpu", weights_only=False)
    if s_infer.dim() == 2:
        s_infer = s_infer.unsqueeze(0)
    elif s_infer.dim() == 3 and s_infer.size(0) != 1:
        s_infer = s_infer[:1]
    s_infer = s_infer.float()

    # Load TextGrids
    src_tg = tgt.io.read_textgrid(sample_data["source_textgrid"])
    tgt_tg = tgt.io.read_textgrid(sample_data["lips_textgrid"])

    # Normalize phonemes
    normalize_textgrid_phonemes(src_tg, TIER_NAME, models['vocab'])
    normalize_textgrid_phonemes(tgt_tg, TIER_NAME, models['vocab'])

    # Store results
    mel_list = []
    phoneme_ids_list = []
    phoneme_texts_list = []
    labels = []

    # Test each configuration
    for warp_type, grid_sample_mode in TEST_CONFIGS:
        config_name = f"{warp_type}_{grid_sample_mode}"
        logger.info(f"Testing: {config_name}")

        # Create transformer
        transformer = SemanticTransformer(
            device=DEVICE,
            verbose=False,
            input_type=warp_type,
            grid_sample_mode=grid_sample_mode,
        )

        # Transform
        if warp_type == "cond":
            # Run LR first, then warp in cond space
            infer_cond_pre = models['len_reg'](
                s_infer.to(DEVICE),
                ylens=torch.LongTensor([max(1, int(s_infer.size(1) * 1.72265625))]).to(DEVICE),
                n_quantizers=3,
                f0=None,
            )[0]
            infer_cond_warped, tgt_duration = transformer.transform(
                x=infer_cond_pre,
                source_textgrid=src_tg,
                target_textgrid=tgt_tg,
                tier_name=TIER_NAME,
            )
            infer_cond = infer_cond_warped
        else:
            # Warp in semantic space first, then run LR
            s_warped, tgt_duration = transformer.transform(
                x=s_infer,
                source_textgrid=src_tg,
                target_textgrid=tgt_tg,
                tier_name=TIER_NAME,
            )
            infer_cond = models['len_reg'](
                s_warped.to(DEVICE),
                ylens=torch.LongTensor([max(1, int(s_warped.size(1) * 1.72265625))]).to(DEVICE),
                n_quantizers=3,
                f0=None,
            )[0]

        # Generate mel with CFM (simplified - no prompt for this test)
        t_gen = infer_cond.size(1)
        x1_mel = models['cfm'].inference(
            infer_cond.to(DEVICE),
            torch.LongTensor([t_gen]).to(DEVICE),
            None,  # No ref_mel for this test
            None,  # No style for this test
            None,  # No f0
            10,    # diffusion_steps
            inference_cfg_rate=0.7,
        ).cpu()

        # Extract frame-level phoneme IDs from target TextGrid
        phoneme_ids, phoneme_texts = textgrid_to_frame_phonemes(
            tgt_tg, TIER_NAME, models['vocab'], x1_mel.shape[-1], MEL_FPS
        )

        mel_list.append(x1_mel)
        phoneme_ids_list.append(phoneme_ids)
        phoneme_texts_list.append(phoneme_texts)
        labels.append(config_name)

        # Save wav (placeholder)
        wav_path = setup_output_dir / f"{sample_stem}_{config_name}.wav"
        mel_to_wav_placeholder(x1_mel, wav_path)

        logger.info(f"  Generated mel shape: {x1_mel.shape}")

    # Add original
    logger.info("Processing original (no transform)")
    infer_cond_orig = models['len_reg'](
        s_infer.to(DEVICE),
        ylens=torch.LongTensor([max(1, int(s_infer.size(1) * 1.72265625))]).to(DEVICE),
        n_quantizers=3,
        f0=None,
    )[0]
    t_gen_orig = infer_cond_orig.size(1)
    x1_mel_orig = models['cfm'].inference(
        infer_cond_orig.to(DEVICE),
        torch.LongTensor([t_gen_orig]).to(DEVICE),
        None, None, None, 10, inference_cfg_rate=0.7,
    ).cpu()

    # Extract frame-level phoneme IDs from source TextGrid
    phoneme_ids_orig, phoneme_texts_orig = textgrid_to_frame_phonemes(
        src_tg, TIER_NAME, models['vocab'], x1_mel_orig.shape[-1], MEL_FPS
    )

    mel_list.append(x1_mel_orig)
    phoneme_ids_list.append(phoneme_ids_orig)
    phoneme_texts_list.append(phoneme_texts_orig)
    labels.append("original")

    # Save original wav
    wav_path_orig = setup_output_dir / f"{sample_stem}_original.wav"
    mel_to_wav_placeholder(x1_mel_orig, wav_path_orig)

    # Save visualization
    vis_path = setup_output_dir / f"{sample_stem}_comparison.png"
    save_multi_mel_visualization(
        mel_list, phoneme_ids_list, phoneme_texts_list, labels,
        str(vis_path), title=f"Sample: {sample_stem}"
    )

    logger.info(f"Completed sample: {sample_stem}")

    # Assertions
    assert len(mel_list) == 5, "Should have 5 mel spectrograms (4 configs + original)"
    for mel in mel_list:
        assert mel.dim() == 3, "Mel should be 3D (B, n_mels, T)"
        assert mel.shape[1] == 80, "Should have 80 mel bins"


if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__, "-v", "-s"])
