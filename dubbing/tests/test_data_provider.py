import sys
from pathlib import Path
import shutil

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
        mse_threshold = 4
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
        print(batch["cond_mel"].shape)  # stretched mel
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
    cond_mel = batch["cond_mel"]  # (B, 80, T)
    x1 = batch["x1"] # (B, 80, T)
    xmean = batch["x_mean"][:, None, None]  # (B, 1, 1)
    xstd = batch["x_std"][:, None, None]   # (B, 1, 1
    # 测试中间态，模拟 CFM 可能对 cond_mel 进行的处理，例如加噪声、时间拉伸等，看看对音质的影响
    xz = cond_mel + 0.1 * torch.randn_like(cond_mel)  # 模拟加噪声
    t = 0.2
    xt = (1-t) * cond_mel + t * x1  # 模拟时间拉伸后的特征
    
    # 做 reverse 归一化
    cond_mel = cond_mel * xstd + xmean
    x1 = x1 * xstd + xmean
    xz = xz * xstd + xmean
    xt = xt * xstd + xmean
    with torch.no_grad():
        wav_x0 = vocoder(cond_mel[max_mse_index:max_mse_index+1].cpu())  # input: (1, 80, T) -> output: (1, 1, T)
        wav_x1 = vocoder(x1[max_mse_index:max_mse_index+1].cpu())
        wav_xz = vocoder(xz[max_mse_index:max_mse_index+1].cpu())
        wav_xt = vocoder(xt[max_mse_index:max_mse_index+1].cpu())
    print(wav_x0.shape)
    print(wav_x1.shape)
    print(wav_xz.shape)
    print(wav_xt.shape) 
    # 可以使用 torchaudio 保存 wav_x0, wav_x1 到文件，检查音质
    import torchaudio
    torchaudio.save("test_x0.wav", wav_x0.squeeze(0), 22050)
    torchaudio.save("test_x1.wav", wav_x1.squeeze(0), 22050)
    torchaudio.save("test_xz.wav", wav_xz.squeeze(0), 22050)
    torchaudio.save("test_xt.wav", wav_xt.squeeze(0), 22050)
    
def test_with_testset(output_dir):
    """
    output testset wavs and textgrid.
    """
    class Args:
        data = "cfm_phase1"
        data_root = "/data2/ruixin/datasets/MELD_gen_pairs"
        train_split_ratio = 0.9
        filter_by_mse = True
        mse_threshold = 4
        tier_name = "phones"
        phoneme_map_path = "dubbing/modules/english_us_arpa_300.json"
        num_workers = 4
        batch_size = 8
        seed = 2026
        
    args = Args()
    test_data, test_loader = data_provider(args, "test")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_to_textgrid = {
        sample.pair_key: (sample.r1_tg, sample.r2_tg)
        for sample in test_data.samples
    }
    
    import bigvgan
    import torch
    import torchaudio
    vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x")
    vocoder.eval()
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 16:
            break

        cond_mel = batch["cond_mel"]
        x1 = batch["x1"]
        xmean = batch["x_mean"][:, None, None]
        xstd = batch["x_std"][:, None, None]
        x_lens = batch["x_lens"]

        cond_mel_denorm = cond_mel * xstd + xmean
        x1_denorm = x1 * xstd + xmean

        with torch.no_grad():
            for sample_idx, pair_key in enumerate(batch["pair_key"]):
                t = int(x_lens[sample_idx].item())
                cond_i = cond_mel_denorm[sample_idx:sample_idx + 1, :, :t].cpu()
                x1_i = x1_denorm[sample_idx:sample_idx + 1, :, :t].cpu()

                wav_cond = vocoder(cond_i)
                wav_x1 = vocoder(x1_i)

                safe_key = str(pair_key).replace("/", "_")
                prefix = f"batch{batch_idx:02d}_idx{sample_idx:02d}_{safe_key}"
                torchaudio.save(str(output_dir / f"{prefix}_cond.wav"), wav_cond.squeeze(0), 22050)
                torchaudio.save(str(output_dir / f"{prefix}_x1.wav"), wav_x1.squeeze(0), 22050)

                tg_pair = pair_to_textgrid.get(pair_key)
                if tg_pair is not None:
                    r1_tg, r2_tg = tg_pair
                    if Path(r1_tg).exists():
                        shutil.copy2(r1_tg, output_dir / f"{prefix}_r1.TextGrid")
                    if Path(r2_tg).exists():
                        shutil.copy2(r2_tg, output_dir / f"{prefix}_r2.TextGrid")
        
    
    
    

if __name__ == "__main__":
    # test_with_args()
    test_with_testset("./test_output")