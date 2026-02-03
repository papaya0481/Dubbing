from indextts.infer_v2 import IndexTTS2

# Initialize the model
# Example: Multiple text segments, each with different emotions
texts = ["Hello, nice to meet you.",
        "But I'm very angry now!",
        "I hope we can get along peacefully.",

]
emotions = [
    "happy: cheerful, energetic, bright confident tone",
    "angry: very irritated, tense, louder and faster",
    "neutral: peaceful, soft, steady rhythmal",
]

# Concatenate text and emotion labels with | separator
text = "|".join(texts)  # "Hello, nice to meet you|But I'm very angry now|I hope we can get along peacefully"
emo_text = "|".join(emotions)  # "happy|angry|neutral"

# Initialize the model
tts = IndexTTS2(
    model_dir="checkpoints",  # Model checkpoint directory
    cfg_path="checkpoints/config.yaml",  # Configuration file path
    is_fp16=False,
)
# Generate
output = tts.infer(
    spk_audio_prompt="voices/test_15_0.mp3",  # Speaker reference audio
    text=text,  # Multiple text segments separated by |
    output_path="attention_results/attn_tests/attention_fs_topk_stw.wav",  # Output path
    method="hmm",
    emo_audio_prompt=None,  # Mode 3 doesn't need emotion reference audio
    emo_alpha=0,  # Mode 3 doesn't need emotion weight
    emo_vector=None,  # Mode 3 doesn't need emotion vector
    use_emo_text=True,  # Enable emotion text labels
    emo_text=emo_text,  # Emotion label sequence separated by |
    use_random=False,  # Whether to use random emotions
    max_text_tokens_per_sentence=150,  # Maximum tokens per sentence
    do_sample=True,  # Sampling switch
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    length_penalty=0,
    num_beams=3,
    repetition_penalty=10.0,
    max_mel_tokens=850  # Maximum mel tokens
)