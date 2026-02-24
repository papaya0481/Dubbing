export CUDA_VISIBLE_DEVICES=2

python mel_convert/generate/gen.py \
    --dataset-option dialog \
    --spk-audio-prompt /data2/ruixin/datasets/MELD_raw_audio \
    --output-dir /data2/ruixin/datasets/MELD_gen_pairs/dialog/ost