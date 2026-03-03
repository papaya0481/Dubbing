export CUDA_VISIBLE_DEVICES=1

accelerate launch dubbing/run.py \
    --is_training 1 \
    --train_epochs 100 \
    --model_id cfm_phase1 \
    --model LipSyncCFM \
    --depth 12 \
    --inference_cfg_rate 0.7 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --log_level DEBUG \
    --early_stop_patience 20 \
    --data_root /data2/ruixin/datasets/MELD_gen_pairs
