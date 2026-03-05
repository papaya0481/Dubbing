import sys
from pathlib import Path

import logging
logging.getLogger().setLevel(logging.WARNING)

project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))
    
import torch
import numpy as np

from modules.mel_strech.mel_transform import GlobalWarpTransformer


def save_mel_phoneme_visualization(
    mel,
    phoneme_ids,
    transformer,
    image_path: str = "test_stretched_mel_with_frame_phonemes.png",
):
    import matplotlib.pyplot as plt

    stretched_mel = mel.detach().cpu().squeeze(0).numpy()   # [n_mels, T]
    ph_tensor = phoneme_ids.detach().cpu()                   # [T] torch.Tensor
    phoneme_ids_np = ph_tensor.numpy().astype(int)           # [T] numpy
    frame_phonemes = transformer._reverse_phoneme_mapping(ph_tensor)
    total_frames = stretched_mel.shape[-1]
    phoneme_ids = phoneme_ids_np

    # 强制音素序列长度与 mel 帧数严格一致
    cur_len = len(frame_phonemes)
    if cur_len < total_frames:
        if cur_len == 0:
            frame_phonemes = ["<eps>"] * total_frames
            phoneme_ids = np.zeros(total_frames, dtype=int)
        else:
            pad_n = total_frames - cur_len
            frame_phonemes = frame_phonemes + [frame_phonemes[-1]] * pad_n
            phoneme_ids = np.pad(phoneme_ids, (0, pad_n), mode="edge")
    elif cur_len > total_frames:
        frame_phonemes = frame_phonemes[:total_frames]
        phoneme_ids = phoneme_ids[:total_frames]

    # 控制图宽，避免语句过长导致图过宽
    fig_width = min(24, max(12, total_frames * 0.08))

    # 用 GridSpec 单独划出一列给 colorbar，保证 mel 图和音素图绘图区等宽
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, 8))
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[4, 1],
        width_ratios=[0.97, 0.03],
        hspace=0.35,
        wspace=0.05,
    )
    ax_mel = fig.add_subplot(gs[0, 0])
    ax_ph  = fig.add_subplot(gs[1, 0], sharex=ax_mel)
    ax_cbar = fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 1]).set_visible(False)  # 占位，保证宽度一致

    mel_img = ax_mel.imshow(
        stretched_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        extent=[-0.5, total_frames - 0.5, -0.5, stretched_mel.shape[0] - 0.5],
    )
    ax_mel.set_title("Stretched Mel Spectrogram")
    ax_mel.set_ylabel("Mel Bins")
    cbar = fig.colorbar(mel_img, cax=ax_cbar)
    cbar.set_label("Log-Mel")

    ax_ph.imshow(
        np.expand_dims(phoneme_ids, axis=0),
        aspect="auto",
        interpolation="nearest",
        cmap="tab20",
        extent=[-0.5, total_frames - 0.5, 0.0, 1.0],
    )
    ax_ph.set_yticks([])
    ax_ph.set_ylabel("Phoneme")
    ax_ph.set_xlabel("Frame Index")
    ax_ph.set_title("Per-frame Phoneme Annotation")

    # 强制上下图 x 轴可视范围完全一致
    x_left, x_right = -0.5, total_frames - 0.5
    ax_mel.set_xlim(x_left, x_right)
    ax_ph.set_xlim(x_left, x_right)

    # 按连续相同音素分段居中标注（比逐帧标注更清晰）
    segments = []
    seg_start = 0
    for idx in range(1, total_frames + 1):
        if idx == total_frames or frame_phonemes[idx] != frame_phonemes[seg_start]:
            segments.append((seg_start, idx - 1, frame_phonemes[seg_start]))
            seg_start = idx

    label_fontsize = 10 if total_frames <= 180 else 8
    for start, end, ph in segments:
        center = (start + end) / 2.0
        ax_ph.text(
            center,
            0.5,
            str(ph),
            ha="center",
            va="center",
            rotation=0,
            fontsize=label_fontsize,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.6},
        )

    tick_step = max(1, total_frames // 20)
    xticks = np.arange(0, total_frames, tick_step)
    ax_ph.set_xticks(xticks)
    ax_ph.set_xticklabels([str(i) for i in xticks], rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close(fig)

    # with open(tsv_path, "w", encoding="utf-8") as f:
    #     f.write("frame\tphoneme_id\tphoneme\n")
    #     for frame_idx in range(total_frames):
    #         f.write(f"{frame_idx}\t{int(phoneme_ids[frame_idx])}\t{frame_phonemes[frame_idx]}\n")

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # transformer = GlobalWarpTransformer(device=device, verbose=True)
    
    # transformer.transform(
    #     source_audio_path="mel_convert/test/test_short1_1.wav",
    #     source_textgrid_path="mel_convert/test/aligned/test_short1_1.TextGrid",
    #     target_textgrid_path="mel_convert/test/aligned/test_short_gt.TextGrid",
    #     target_audio_path="mel_convert/test/test_short_gt.wav",
    #     output_path="output_.wav",
    #     tier_name="phones"
    # )
    
    # test for 
    SOURCE_DIR = "sent_emo"
    SOURCE_PATH = "test_dia168_utt3"
    
    import tgt
    source_tg = tgt.io.read_textgrid(f"/data2/ruixin/datasets/MELD_gen_pairs/{SOURCE_DIR}/aligned/{SOURCE_PATH}_r1.TextGrid")
    target_tg = tgt.io.read_textgrid(f"/data2/ruixin/datasets/MELD_gen_pairs/{SOURCE_DIR}/aligned/{SOURCE_PATH}_r2.TextGrid")
    source_wav = f"/data2/ruixin/datasets/MELD_gen_pairs/{SOURCE_DIR}/ost/{SOURCE_PATH}_r1.wav"
    transformer = GlobalWarpTransformer(use_vocoder=True, device="cpu", verbose=True)
    
    wav, sr = transformer.load_audio(source_wav)
    from modules.mel_strech.meldataset import get_mel_spectrogram
    mel_sorce = get_mel_spectrogram(wav, transformer.h)
    
    out = transformer.transform_mel_with_path(
        source_mel=mel_sorce,
        source_textgrid=source_tg,
        target_textgrid=target_tg,
        tier_name="phones",
    )
    
    print(out)
    print(out[0].shape, out[1].shape)
    print(transformer._reverse_phoneme_mapping(out[1]))
    print(source_tg.get_tier_by_name("phones"))
    print(target_tg.get_tier_by_name("phones"))
    print(source_tg.get_tier_by_name("words"))
    print(target_tg.get_tier_by_name("words"))
    
    # 可视化 mel spectrogram 与帧级音素标注
    save_mel_phoneme_visualization(out[0], out[1], transformer)

    # 源音频恒等 warp（source_tg 同时作为 source 和 target），取得帧级音素用于可视化
    src_out = transformer.transform_mel_with_path(
        source_mel=mel_sorce,
        source_textgrid=source_tg,
        target_textgrid=source_tg,
        tier_name="phones",
    )
    save_mel_phoneme_visualization(
        src_out[0], src_out[1], transformer,
        image_path="test_source_mel_with_frame_phonemes.png",
    )

    
    
    # 使用 vocoder 将 out[0] 转成音频，保存到文件   
    wav_out = transformer.model(out[0]).cpu().squeeze(0)
    import torchaudio
    torchaudio.save("test_out.wav", wav_out, sr)
    
    # 检查时长是否和 target_tg 的 phones tier 对齐
    print(target_tg.get_tier_by_name("phones").end_time)
    wav_out_len = wav_out.shape[-1]
    print(f"Output wav length (samples): {wav_out_len}, duration (s): {wav_out_len/sr:.2f}s")
    
    # 测试 words tier 的对齐
    out2 = transformer.transform_mel_with_path(
        source_mel=mel_sorce,
        source_textgrid=source_tg,
        target_textgrid=target_tg,
        tier_name="words",
    )
    
    wav_out2 = transformer.model(out2[0]).cpu().squeeze(0)
    torchaudio.save("test_out_words.wav", wav_out2, sr)