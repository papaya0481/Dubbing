export CUDA_VISIBLE_DEVICES=2

python index-tts2/batch_gen2_timing.py \
    --data_file index-tts2/results/phase2_chinese_fixed.json \
    --dataset_dir ./ESD/ESD \
    --verbose \
    --save_dir index-tts2/results/rtf/chn_baseline_msa