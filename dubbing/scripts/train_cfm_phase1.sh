export CUDA_VISIBLE_DEVICES=3

python -m pdb dubbing/run.py \
    --is_training 1 \
    --train_epochs 100 \
    --model_id cfm_phase1 \
    --model LipSyncCFM \
    --inference_cfg_rate 0.9 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --log_level DEBUG \
    --data_root /data2/ruixin/datasets/MELD_gen_pairs
