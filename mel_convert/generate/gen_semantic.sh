export CUDA_VISIBLE_DEVICES=2

python mel_convert/generate/gen_semantic_stretch.py \
    --metadata-csv /data2/ruixin/datasets/MELD_gen_pairs_semanti/origin/dialog/generation_metadata.csv \
    --output-dir   /data2/ruixin/datasets/MELD_gen_pairs_semanti/semantic_stretch/dialog \
    --model-dir    /data2/ruixin/index-tts2/checkpoints \
    --index-tts-root /home/ruixin/Dubbing/dubbing