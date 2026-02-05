from indextts.infer_v2 import IndexTTS2


def seconds_to_mel_tokens(seconds, mel_to_sec_ratio=0.02):
    """
    将秒数转换为 mel tokens
    
    Args:
        seconds: float or list of floats, 目标秒数
        mel_to_sec_ratio: float, 转换比例
            - 默认 0.02 表示: 1 mel token ≈ 0.02 秒
            - 换算: 50 mel tokens ≈ 1 秒
    
    Returns:
        int or list of ints, mel token 数量
    
        >>> seconds_to_mel_tokens(1.5)
        75
        >>> seconds_to_mel_tokens([1.0, 2.0, 1.5])
        [50, 100, 75]
    """
    if isinstance(seconds, (list, tuple)):
        return [int(s / mel_to_sec_ratio) for s in seconds]
    else:
        return int(seconds / mel_to_sec_ratio)


if __name__ == "__main__":
    
    checkpoint_path = "/data2/ruixin/index-tts2/checkpoints"
    
    tts = IndexTTS2(
        model_dir=checkpoint_path,
        cfg_path=f"{checkpoint_path}/config.yaml",
        is_fp16=False
    )
    
    texts = ['I left my guitar in their apartment.', 'Well you can let me in later.']
    
    emotions = [
        "surprise",
        "neutral",
    ]
    
    emotion_prompt_path = "/data2/ruixin/datasets/MELD_clips/audios/ost/dev_sample_5.wav"
    
    text = "|".join(texts)
    emo_text = "|".join(emotions)
    
    # 设置每段的秒数
    target_seconds = [1.5, 1.5, 3]
    
    # 转换为 mel tokens（每段）
    target_duration_tokens = seconds_to_mel_tokens(target_seconds)

    target_duration_tokens = None
    output = tts.infer(
        spk_audio_prompt=emotion_prompt_path,
        text=text,
        output_path="test_short1_1.wav",
        style_prompt=None,
        emo_audio_prompt=None,
        emo_alpha=0,
        use_emo_text=True,
        emo_text=emo_text,
        use_random=False,
        verbose=True,
        emo_vector=None,
        
        # duration_targets=cumulative_targets,
        target_duration_tokens=target_duration_tokens,
        method="hmm",                          # 使用HMM段切换
        save_attention_maps=False,
        
        # 生成参数
        max_text_tokens_per_sentence=200,
        do_sample=True,
        top_p=0.8,
        top_k=30,
        temperature=0.8,
        length_penalty=0,
        num_beams=2,
        repetition_penalty=10.0,
        max_mel_tokens=2000
    )
    
    print(f"\nGenerated audio saved to: {output}")
