export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python index-tts2/batch_gen2_ablation_spc_input.py \
    --data_file index-tts2/results/phase2_english_fixed.json \
    --dataset_dir ./ESD/ESD \
    --max_samples 100 \
    --save_dir index-tts2/results/ablation/eng_max_head2