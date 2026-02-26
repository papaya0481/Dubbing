export CUDA_VISIBLE_DEVICES=3

python dubbing/run.py \
    --is_training 1 \
    --train_epochs 10 \
    --model_id cfm_phase1 \
    --model LipSyncCFM \
    --lr_end_factor 0.5 \
    --data_root /data2/ruixin/datasets/MELD_gen_pairs
