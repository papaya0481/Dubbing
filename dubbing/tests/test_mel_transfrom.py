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
    
    from transformers import AutoConfig
    print(AutoConfig.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x"))
    
    # test for 
    
    import tgt
    source_tg = tgt.io.read_textgrid("mel_convert/test/aligned/test_short1_1.TextGrid")
    target_tg = tgt.io.read_textgrid("mel_convert/test/aligned/test_short_gt.TextGrid")
