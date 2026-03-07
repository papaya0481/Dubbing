"""
test_mel_align.py
=================
输入一个音频文件和对应文本，使用 MFAAligner 做强制对齐，
然后画出梅尔频谱图（含帧级音素标注）。

用法：
    python dubbing/tests/test_mel_align.py \\
        --wav  /path/to/audio.wav \\
        --text "Hello world" \\
        [--output test_mel_align.png]
"""

import sys
from pathlib import Path

import torch
import torchaudio

project_dubbing_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dubbing_root))
sys.path.insert(0, str(project_dubbing_root / "modules"))

from modules.mfa_alinger import MFAAligner
from modules.mel_strech.mel_transform import GlobalWarpTransformer
from modules.mel_strech.meldataset import get_mel_spectrogram


# ---------------------------------------------------------------------------
# 可视化（与 test_mel_transfrom.py 一致）
# ---------------------------------------------------------------------------

def save_mel_phoneme_visualization(mel, phoneme_ids, transformer, image_path, title=""):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    stretched_mel = mel.detach().cpu().squeeze(0).numpy()
    ph_tensor = phoneme_ids.detach().cpu()
    phoneme_ids_np = ph_tensor.numpy().astype(int)
    frame_phonemes = transformer._reverse_phoneme_mapping(ph_tensor)
    total_frames = stretched_mel.shape[-1]
    phoneme_ids_arr = phoneme_ids_np

    cur_len = len(frame_phonemes)
    if cur_len < total_frames:
        pad_n = total_frames - cur_len
        frame_phonemes = frame_phonemes + [frame_phonemes[-1] if cur_len else "<eps>"] * pad_n
        phoneme_ids_arr = np.pad(phoneme_ids_arr, (0, pad_n), mode="edge")
    elif cur_len > total_frames:
        frame_phonemes = frame_phonemes[:total_frames]
        phoneme_ids_arr = phoneme_ids_arr[:total_frames]

    fig_width = min(24, max(12, total_frames * 0.08))
    fig = plt.figure(figsize=(fig_width, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 1],
                  width_ratios=[0.97, 0.03], hspace=0.35, wspace=0.05)
    ax_mel  = fig.add_subplot(gs[0, 0])
    ax_ph   = fig.add_subplot(gs[1, 0], sharex=ax_mel)
    ax_cbar = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).set_visible(False)

    mel_img = ax_mel.imshow(
        stretched_mel, origin="lower", aspect="auto",
        interpolation="nearest", cmap="magma",
        extent=[-0.5, total_frames - 0.5, -0.5, stretched_mel.shape[0] - 0.5],
    )
    ax_mel.set_title(f"Mel Spectrogram — {title}" if title else "Mel Spectrogram with MFA Alignment")
    ax_mel.set_ylabel("Mel Bins")
    fig.colorbar(mel_img, cax=ax_cbar).set_label("Log-Mel")

    ax_ph.imshow(
        np.expand_dims(phoneme_ids_arr, axis=0),
        aspect="auto", interpolation="nearest", cmap="tab20",
        extent=[-0.5, total_frames - 0.5, 0.0, 1.0],
    )
    ax_ph.set_yticks([])
    ax_ph.set_ylabel("Phoneme")
    ax_ph.set_xlabel("Frame Index")
    ax_ph.set_title("Per-frame Phoneme (MFA)")

    x_left, x_right = -0.5, total_frames - 0.5
    ax_mel.set_xlim(x_left, x_right)
    ax_ph.set_xlim(x_left, x_right)

    segments, seg_start = [], 0
    for idx in range(1, total_frames + 1):
        if idx == total_frames or frame_phonemes[idx] != frame_phonemes[seg_start]:
            segments.append((seg_start, idx - 1, frame_phonemes[seg_start]))
            seg_start = idx

    label_fontsize = 10 if total_frames <= 180 else 8
    for start, end, ph in segments:
        ax_ph.text(
            (start + end) / 2.0, 0.5, str(ph),
            ha="center", va="center", fontsize=label_fontsize, color="black",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.6},
        )

    import numpy as np
    tick_step = max(1, total_frames // 20)
    xticks = np.arange(0, total_frames, tick_step)
    ax_ph.set_xticks(xticks)
    ax_ph.set_xticklabels([str(i) for i in xticks], fontsize=9)

    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close(fig)
    print(f"[可视化] 保存至 {image_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- 配置输入 ----
    WAV_PATH    = "/home/ruixin/Dubbing/test_output/test_semantic_warp_mid.wav"
    TEXT        = "What-You're not serious. I mean she's a very nice woman, but there is no way we can take eight weeks of her. She'll drive us totally crazy."
    BEAM        = 20
    RETRY_BEAM  = 200

    wav_path = Path(WAV_PATH)
    out_dir  = Path(__file__).resolve().parents[2] / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT   = str(out_dir / (wav_path.stem + ".png"))
    assert wav_path.exists(), f"音频不存在: {wav_path}"
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_mono = wav.squeeze(0)   # (N,)

    # ---- MFA 对齐 ----
    print(f"[MFA] 对齐文本: {TEXT!r}")
    aligner = MFAAligner(beam=BEAM, retry_beam=RETRY_BEAM)
    tg, phone_groups = aligner.align_one_wav(
        wavs=wav_mono,
        sampling_rate=sr,
        text=TEXT,
        return_textgrid=True,
    )
    print(f"[MFA] 完成，phone tier: {tg.get_tier_by_name('phones')}")
    print(f"[MFA] word tier:  {tg.get_tier_by_name('words')}")

    # ---- 计算梅尔频谱 ----
    transformer = GlobalWarpTransformer(use_vocoder=False, device="cpu", verbose=False)

    # GlobalWarpTransformer.load_audio 返回 (wav_tensor, sr)，这里直接复用已加载的 wav
    # get_mel_spectrogram 需要 (1, N) 或 (N,)
    wav_for_mel = wav_mono.unsqueeze(0)   # (1, N)
    mel = get_mel_spectrogram(wav_for_mel, transformer.h)   # (1, n_mels, T)

    # ---- 将 TextGrid 的音素 tier 映射到帧级 phoneme_ids ----
    # 借用 transform_mel_with_path 的 identity warp（source == target）获得帧级标注
    mel_out, phoneme_ids = transformer.transform_mel_with_path(
        source_mel=mel,
        source_textgrid=tg,
        target_textgrid=tg,   # 恒等 warp，只取帧级标注
        tier_name="phones",
    )

    # ---- 可视化 ----
    save_mel_phoneme_visualization(mel_out, phoneme_ids, transformer, OUTPUT, title=wav_path.name)


if __name__ == "__main__":
    main()
