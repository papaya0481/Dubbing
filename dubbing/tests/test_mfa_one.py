import sys
from pathlib import Path

project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))

import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from indextts.inferDub import IndexTTS2ForDub
import torchaudio
import torch



def test_mfa_aligner_with_dub():
    checkpoint_path = "/data2/ruixin/index-tts2/checkpoints"
    model = IndexTTS2ForDub(
        model_dir=checkpoint_path,
        cfg_path=f"{checkpoint_path}/config.yaml",
        is_fp16=False
    )
    
    # 等待现存分配稳定
    from modules.mfa_alinger import MFAAligner
    spk_prompt = "/data2/ruixin/ted-tts/AllInferenceResults/ESD/0001/Angry/0001_000351.wav"

    texts = ['I left my guitar in their apartment.', 'Well you can let me in later.']

    emotions = [
        "surprise",
        "angry",
    ]

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=texts,
        output_path=None,
        use_emo_text=True,
        emo_text=emotions,
        emo_alpha=1.4,
        verbose=True,
        
        method="hmm",
        max_text_tokens_per_sentence=200,
        do_sample=True,
        top_p=0.8,
        top_k=30,
        temperature=0.8,
        length_penalty=0,
        num_beams=3,
        repetition_penalty=10.0,
        max_mel_tokens=2000,
        return_stats=True,
    )
    
    print("Inference result:", result)
    
    # save wav to file
    wavs = result.wavs.type(torch.int16)
    torchaudio.save("dubbed_audio.wav", wavs, result.sampling_rate)
    
    import librosa
    wavs_reloaded = librosa.load("dubbed_audio.wav", sr=16000, mono=False)[0]
    print(f"reloaded wav: {wavs_reloaded}, ")
    print(f"reloaded wav shape: {wavs_reloaded.shape}, ")
    print(f"original wav shape: {wavs.shape}, ")
    
    aligner = MFAAligner()
    texts = " ".join(texts)
    import time
    start_time = time.time()
    align_result = aligner.align_one_wav(
        text=texts,
        wavs=result.wavs,
        text_file_path="mel_convert/test/test_short1_1.txt"
    )
    end_time = time.time()
    print(f"Time taken for alignment: {end_time - start_time:.2f} seconds")
    print("Alignment result:", align_result)

if __name__ == "__main__":
    test_mfa_aligner_with_dub()