import json
import logging
import os
import sys
import threading
import time

import warnings
from hashlib import sha256

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd


def str_to_int(s: str) -> int:
    # 1. 编码为 bytes
    b = s.encode('utf-8')
    # 2. 计算 sha256
    h = sha256(b).hexdigest()
    # 3. 转为 int
    return int(h, 16)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts import infer
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto
from modelscope.hub import api

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),is_fp16=cmd_args.is_fp16)


def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight, method,
               emo_text,emo_random,
               vec1=0, vec2=0, vec3=0, vec4=0, vec5=0, vec6=0, vec7=0, vec8=0,
               max_text_tokens_per_sentence=1000, output_path=None,
               do_sample = True, top_p = 0.8, top_k = 30, temperature = 0.8, length_penalty = 0, num_beams = 3, repetition_penalty = 10, max_mel_tokens = 150, progress=gr.Progress()):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
        "method": method,
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning(i18n("情感向量之和不能超过1.5，请调整后重试。"))
            return
    else:
        vec = None

    print(f"Emo control mode:{emo_control_method},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                       **kwargs)
    return output

with open("index-tts2/results/test.json","r",encoding="utf-8") as f:
    input_list = json.load(f)

results = []

spk_audios = [
    "voices/test_0_1.mp3",
    "voices/test_12_0.mp3",
    "voices/test_15_0.mp3",
    "voices/test_25_14.mp3",
    "voices/train_0_0.mp3",
    "voices/train_0_5.mp3",
]

output_path = "index-tts2/results/test/hmm"

if not os.path.exists(output_path):
    os.makedirs(output_path)


if os.path.exists(os.path.join(output_path, "inference_results_with_wav.json")):
    with open(os.path.join(output_path, "inference_results_with_wav.json"),"r",encoding="utf-8") as f:
        results = json.load(f)
for id, item in enumerate(input_list):
    if any(r["id"] == id for r in results):
        print(f"Skipping already processed item {id}")
        continue
    try:
        texts = []
        emotions = []
        emo_map = {}
        for i in item["emo_seq"]:
            k, v = list(i.keys())[0], list(i.values())[0]
            emo_map[k.lower().strip()] = v
        if "output" not in item or "mappings" not in item["output"]:
            print(f"Item {id} missing output mappings, skipping.")
            continue
        for line in item["output"]["mappings"]:
            texts.append(line["segment"])
            emotions.append(
                line["emotion"] + ": " + emo_map[line["emotion"].lower().strip()] 
                if line["emotion"].lower().strip() in emo_map else line["emotion"].lower().strip()
            )
        text = "|".join(texts)
        emo_text = "|".join(emotions)
        spk_audio = spk_audios[(len(text) + 2) % len(spk_audios)]
        print(f"Using speaker audio: {spk_audio}")
        item["spk_audio"] = spk_audio
        item["id"] = id
        output_wav_path = os.path.join(output_path, f"output_{id}.wav")
        print(f"Generating {output_wav_path} ...")
        print(f"Text: {text}")
        print(f"Emo Text: {emo_text}")
        gen_single(
            emo_control_method=3,
            prompt=spk_audio,
            text=text,
            emo_ref_path=None,
            emo_weight=0,
            method="hmm",
            emo_text=emo_text,
            emo_random=False,
            max_text_tokens_per_sentence=150,
            output_path=output_wav_path,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            length_penalty=0,
            num_beams=3,
            repetition_penalty=10,
            max_mel_tokens=850
        )
        item["output_wav_path"] = output_wav_path
        results.append(item)
    except Exception as e:
        import traceback
        print(f"Error processing item {id}: {e}")
        traceback.print_exc()
        
    if id % 10 == 0:
        with open(os.path.join(output_path, "inference_results_with_wav.json"),"w",encoding="utf-8") as f:
            json.dump(results,f,ensure_ascii=False,indent=2)

with open(os.path.join(output_path, "inference_results_with_wav.json"),"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
    

    