from indextts.inferDub import IndexTTS2ForDub

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
        "neutral",
    ]

    output_path = "dubbed_audio.wav"

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=texts,
        output_path=output_path,
        use_emo_text=True,
        verbose=True,
        return_stats=True
    )

    print("Inference result:", result)