import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textgrid  # 用于处理 MFA 输出的 TextGrid 文件
import os

# --- 1. 配置路径 ---
# 用户输入一个音频文件夹，一个textgrid文件夹
wav_dir = 'mel_convert/test'
tg_dir = 'mel_convert/test/aligned'
# 输出的图片也存放在一个文件夹里
LEVEL_NAME = 'words'  # 选择对齐的层级名称，通常是 'words' 或 'phones'
output_dir = f'mel_convert/test/images_{LEVEL_NAME}'

# 可选：decode_output.json 路径，设为 None 则不显示 token 标注条
# 例如：decode_json_path = 'decode_output.json'
decode_json_path = 'decode_output.json'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 加载 decode_output.json（如果提供） ---
decode_data = {}
if decode_json_path and os.path.exists(decode_json_path):
    with open(decode_json_path, 'r', encoding='utf-8') as f:
        decode_data = json.load(f)
    print(f"Loaded {decode_json_path}: {len(decode_data)} entries")

# 获取文件列表
wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

print(f"Found {len(wav_files)} wav files in {wav_dir}")

for wav_name in wav_files:
    base_name = os.path.splitext(wav_name)[0]
    tg_name = base_name + '.TextGrid'
    
    wav_file = os.path.join(wav_dir, wav_name)
    tg_file = os.path.join(tg_dir, tg_name)
    
    if not os.path.exists(tg_file):
        print(f"Warning: TextGrid file not found for {wav_name}, skipping...")
        continue
        
    print(f"Processing {base_name}...")

    # --- 2. 生成梅尔频谱 ---
    y, sr = librosa.load(wav_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # --- 3. 读取 TextGrid 文件 ---
    tg = textgrid.TextGrid.fromFile(tg_file)
    target_tier = tg.getFirst(LEVEL_NAME)

    # --- 4. 从 decode_output.json 解析 token 分段 ---
    # token_segments: list of (t_start, t_end, label_str, group_id)
    token_segments = None
    if wav_name in decode_data:
        entry = decode_data[wav_name]
        pieces_raw = entry.get('pieces', [])
        aligned_seq = entry.get('aligned_sequences', [])
        if pieces_raw and aligned_seq:
            # 兼容旧格式（flat list）和新格式（list of lists）
            if pieces_raw and isinstance(pieces_raw[0], list):
                # 新格式：展平，同时记录每个 piece 属于哪个 group
                flat_pieces = []
                piece_group = []  # flat index → group id (0-based)
                for g, sublist in enumerate(pieces_raw):
                    for p in sublist:
                        flat_pieces.append(p)
                        piece_group.append(g)
            else:
                flat_pieces = pieces_raw
                piece_group = [0] * len(flat_pieces)

            # 将索引转为 0-based
            aligned_seq = [v - 1 for v in aligned_seq]
            n = len(aligned_seq)

            # 合并相邻相同索引 → (frame_start, frame_end, piece_text, group_id)
            raw_segments = []
            prev_idx = aligned_seq[0]
            seg_start = 0
            for i in range(1, n):
                if aligned_seq[i] != prev_idx:
                    label = flat_pieces[prev_idx] if prev_idx < len(flat_pieces) else '?'
                    grp   = piece_group[prev_idx] if prev_idx < len(piece_group) else 0
                    raw_segments.append((seg_start, i, label, grp))
                    seg_start = i
                    prev_idx = aligned_seq[i]
            label = flat_pieces[prev_idx] if prev_idx < len(flat_pieces) else '?'
            grp   = piece_group[prev_idx] if prev_idx < len(piece_group) else 0
            raw_segments.append((seg_start, n, label, grp))

            # 按比例换算成秒
            token_segments = [
                (s / n * duration, e / n * duration, lbl, g)
                for s, e, lbl, g in raw_segments
            ]

    # --- 5. 绘图 ---
    if token_segments:
        # 上方细条（token 标注）+ 下方梅尔频谱
        fig = plt.figure(figsize=(14, 6.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.0,
                               figure=fig)
        ax_tokens = fig.add_subplot(gs[0])
        ax_mel    = fig.add_subplot(gs[1], sharex=ax_tokens)
    else:
        fig, ax_mel = plt.subplots(figsize=(14, 5))
        ax_tokens = None

    # A. 梅尔频谱
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel',
                                   fmax=sr/2, cmap='viridis', ax=ax_mel)

    # 用 make_axes_locatable 在 ax_mel 右侧切出 colorbar 区域；
    # 若有 ax_tokens，同样在其右侧切出等宽的不可见占位条，保证两行宽度一致。
    divider_mel = make_axes_locatable(ax_mel)
    cax = divider_mel.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(img, cax=cax, format='%+2.0f dB')

    if ax_tokens is not None:
        divider_tok = make_axes_locatable(ax_tokens)
        cax_tok = divider_tok.append_axes('right', size='2%', pad=0.05)
        cax_tok.set_visible(False)

    # B. TextGrid 对齐信息叠加在梅尔频谱上
    for interval in target_tier:
        if interval.mark:
            ax_mel.vlines(interval.minTime, 0, sr / 2,
                          colors='yellow', linestyles='dashed', alpha=0.7, linewidth=1)
            ax_mel.vlines(interval.maxTime, 0, sr / 2,
                          colors='white', linestyles='dotted', alpha=0.7, linewidth=1)
            center_time = (interval.minTime + interval.maxTime) / 2
            ax_mel.text(center_time, sr / 2 * 0.8, interval.mark,
                        color='white', ha='center', fontsize=7, rotation=90)

    # C. Token 标注条（顶部 x 轴）
    if ax_tokens is not None and token_segments:
        # 每个 group 有两套交替底色（亮/暗），group 间色调不同
        # 调色方案：每组取一个主色调，交替明暗
        GROUP_PALETTE = [
            ('#1a3a5c', '#0d2540'),  # 蓝色系
            ('#3a1a1a', '#250d0d'),  # 红色系
            ('#1a3a1a', '#0d250d'),  # 绿色系
            ('#3a3a1a', '#25250d'),  # 黄色系
            ('#2a1a3a', '#1a0d25'),  # 紫色系
        ]

        ax_tokens.set_xlim(0, duration)
        ax_tokens.set_ylim(0, 1)
        ax_tokens.set_yticks([])
        ax_tokens.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_tokens.set_facecolor('#1e1e2e')

        # 每个 group 内部用奇偶交替，不同 group 用不同色系
        group_alt_counter = {}  # group_id → count of segments seen so far
        for t_start, t_end, label, grp in token_segments:
            palette = GROUP_PALETTE[grp % len(GROUP_PALETTE)]
            k = group_alt_counter.get(grp, 0)
            bg_color = palette[k % 2]
            group_alt_counter[grp] = k + 1

            ax_tokens.axvspan(t_start, t_end, color=bg_color, alpha=1.0)
            ax_tokens.axvline(x=t_start, color='#aaaacc', linewidth=0.8)
            label_clean = label.replace('▁', '').strip() or '·'
            center = (t_start + t_end) / 2
            ax_tokens.text(center, 0.5, label_clean,
                           ha='center', va='center',
                           fontsize=7, color='white', clip_on=True)
        ax_tokens.axvline(x=duration, color='#aaaacc', linewidth=0.8)

        ax_tokens.set_title(
            f'Token Alignment + Mel Spectrogram ({LEVEL_NAME}) — {base_name}',
            fontsize=9, pad=3, loc='left'
        )
    else:
        ax_mel.set_title(
            f'Mel Spectrogram with MFA Alignment ({target_tier.name}) - {base_name}'
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

print("Done.")
