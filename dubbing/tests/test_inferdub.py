import sys
from pathlib import Path

project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))

from indextts.inferDub import IndexTTS2ForDub
import torchaudio
import torch

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/data2/ruixin/index-tts2/checkpoints"
    model = IndexTTS2ForDub(
        model_dir=checkpoint_path,
        cfg_path=f"{checkpoint_path}/config.yaml",
        is_fp16=False
    )
    spk_prompt = "/data2/ruixin/ted-tts/AllInferenceResults/ESD/0001/Angry/0001_000351.wav"

    texts = ['I left my guitar in their apartment.', 'Well you can let me in later.']

    emotions = [
        "surprise",
        "angry",
    ]

    output_path = "dubbed_audio.wav"

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=texts,
        output_path=output_path,
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