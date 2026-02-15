import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import textgrid  # 用于处理 MFA 输出的 TextGrid 文件
import os

# --- 1. 配置路径 ---
# 用户输入一个音频文件夹，一个textgrid文件夹
wav_dir = 'mel_convert/test'
tg_dir = 'mel_convert/test/aligned'
# 输出的图片也存放在一个文件夹里
LEVEL_NAME = 'words'  # 选择对齐的层级名称，通常是 'words' 或 'phones'
output_dir = f'mel_convert/test/images_{LEVEL_NAME}'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

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

    # --- 2. 生成梅尔频谱 (与之前相同) ---
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # --- 3. 读取 TextGrid 文件 ---
    tg = textgrid.TextGrid.fromFile(tg_file)

    # 通常 MFA 的 TextGrid 有两个层级 (tiers)：
    # tier[0] 通常是 "words" (单词层)
    # tier[1] 通常是 "phones" (音素层)
    # 这里我们演示可视化 "phones" 层，你可以根据需要改为 words
    target_tier = tg.getFirst(LEVEL_NAME) 

    # --- 4. 绘图与叠加 ---
    plt.figure(figsize=(14, 5))

    # A. 画底层的梅尔频谱
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=sr/2, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')

    # B. 叠加对齐信息
    # 遍历该层级中的所有区间 (Intervals)
    for interval in target_tier:
        # interval.minTime: 开始时间
        # interval.maxTime: 结束时间
        # interval.mark: 标签内容 (音素或单词)
        
        # 忽略空的标签（有时候 MFA 会产生空隙或 silence 标记为 "" 或 "sp"）
        if interval.mark: 
            # 1. 画垂直分割线 (边界)
            # 起始时间 (用绿色表示)
            plt.vlines(interval.minTime, 0, sr/2, colors='yellow', linestyles='dashed', alpha=0.7, linewidth=1)
            # 结束时间 (用白色表示)
            plt.vlines(interval.maxTime, 0, sr/2, colors='white', linestyles='dotted', alpha=0.7, linewidth=1)
            
            # 2. 添加文本标签 (居中显示)
            center_time = (interval.minTime + interval.maxTime) / 2
            plt.text(center_time, 
                     sr/2 * 0.8,  # 放在频域的 80% 高度位置，避免遮挡主要频谱
                     interval.mark, 
                     color='white', 
                     horizontalalignment='center', 
                     fontsize=7,
                     rotation=90) # 如果音素很密集，可以旋转文字

    plt.title(f'Mel Spectrogram with MFA Alignment ({target_tier.name}) - {base_name}')
    plt.tight_layout()
    
    # 1. 将图片保存而不是展示
    save_path = os.path.join(output_dir, f'{base_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close() # 关闭当前图形，释放内存
    
    print(f"Saved {save_path}")

print("Done.")
