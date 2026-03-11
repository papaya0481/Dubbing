export CUDA_VISIBLE_DEVICES=1

python mel_convert/generate/gen_semantic.py \
    --csv /data2/ruixin/datasets/MELD_raw/metadata.csv \
    --output-dir /data2/ruixin/datasets/flow_dataset/MELD_semantic \
    --model-dir /data2/ruixin/index-tts2/checkpoints \
    --gpus 0 \
    --num-process 2