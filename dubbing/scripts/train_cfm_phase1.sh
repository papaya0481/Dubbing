export CUDA_VISIBLE_DEVICES=3

python -m pdb dubbing/run.py \
    --is_training 1 \
    --model_id cfm_phase1 \
    --model LipSyncCFM \
    --data_root /data2/ruixin/datasets/MELD_gen_pairs
