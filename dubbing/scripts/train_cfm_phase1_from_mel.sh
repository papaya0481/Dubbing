export CUDA_VISIBLE_DEVICES=1

accelerate launch dubbing/run.py \
    --is_training 1 \
    --train_epochs 100 \
    --model_id cfm_phase1 \
    --model LipSyncCFM \
    --inference_cfg_rate 1.5 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --log_level DEBUG \
    --data_root /data2/ruixin/datasets/MELD_gen_pairs
