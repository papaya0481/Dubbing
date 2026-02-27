import sys
from pathlib import Path

import logging
logging.getLogger().setLevel(logging.WARNING)

project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))
    
import torch

from modules.mel_strech.mel_transform import GlobalWarpTransformer

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
    
    import tgt
    source_tg = tgt.io.read_textgrid("/home/ruixin/Dubbing/mel_convert/test/aligned/test_short1_1.TextGrid", include_empty_intervals=True)
    target_tg = tgt.io.read_textgrid("/home/ruixin/Dubbing/mel_convert/test/aligned/test_short_gt.TextGrid", include_empty_intervals=True)
    source_wav = "/home/ruixin/Dubbing/mel_convert/test/test_short1_1.wav"
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
    
    # 计算对齐之间的mse
    
    
    # 使用 vocoder 将 out[0] 转成音频，保存到文件   
    wav_out = transformer.model(out[0]).cpu().squeeze(0)
    import torchaudio
    torchaudio.save("test_out.wav", wav_out, sr)
    
    # 检查时长是否和 target_tg 的 phones tier 对齐
    print(target_tg.get_tier_by_name("phones").end_time)
    wav_out_len = wav_out.shape[-1]
    print(f"Output wav length (samples): {wav_out_len}, duration (s): {wav_out_len/sr:.2f}s")