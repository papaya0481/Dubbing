export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1   # 让 device-side assert 立即在真实出错位置暴露

python mel_convert/generate/gen_once.py \
    --dataset-option dialog \
    --spk-audio-prompt /data2/ruixin/datasets/MELD_raw_audio \
    --output-dir /data2/ruixin/datasets/MELD_gen_pairs_semanti/dialog