export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --multi_gpu --num_processes=2 dubbing/run.py \
    --config dubbing/configs/default_cfm_index.yaml