import sys
from pathlib import Path

import logging
logging.getLogger().setLevel(logging.WARNING)

project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))

from data_provider.data_factory import data_provider
import numpy as np

def test_with_args():
    class Args:
        data = "cfm_phase1"
        data_root = "/data2/ruixin/datasets/MELD_gen_pairs"
        train_split_ratio = 0.9
        filter_by_mse = True
        mse_threshold = 8
        tier_name = "phones"
        phoneme_map_path = "dubbing/modules/english_us_arpa_300.json"
        num_workers = 4
        batch_size = 8
        seed = 2026

    args = Args()
    train_data, train_loader = data_provider(args, "train")
    
    # 取出一个 batch，检查数据格式
    for batch in train_loader:
        print(batch)
        print(batch["x0"].shape)  # stretched mel
        print(batch["x1"].shape)  # clean mel
        print(batch["phoneme_ids"][0])  # phoneme ids
        
        # 找到mse最大的样本
        mse_values = np.array(batch["mse"])
        max_mse_index = mse_values.argmax()
        print(f"Max MSE: {mse_values[max_mse_index]}, Pair Key: {batch['pair_key'][max_mse_index]}, Phoneme IDs: {batch['phoneme_ids'][max_mse_index]}")
        
        break

    import bigvgan
    import torch
    vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x")
    # 测试 vocoder 将x0, x1 转成音频
    x0 = batch["x0"]  # (B, 80, T)
    x1 = batch["x1"]  # (B, 80, T)
    with torch.no_grad():
        wav_x0 = vocoder(x0[max_mse_index:max_mse_index+1].cpu())  # input: (1, 80, T) -> output: (1, 1, T)
        wav_x1 = vocoder(x1[max_mse_index:max_mse_index+1].cpu())
    print(wav_x0.shape)
    print(wav_x1.shape)
    # 可以使用 torchaudio 保存 wav_x0, wav_x1 到文件，检查音质
    import torchaudio
    torchaudio.save("test_x0.wav", wav_x0.squeeze(0), 22050)
    torchaudio.save("test_x1.wav", wav_x1.squeeze(0), 22050)
    
    

if __name__ == "__main__":
    test_with_args()