export CUDA_VISIBLE_DEVICES=0

src=(
    /data2/ruixin/datasets/v2c_origin
    /data2/ruixin/downloads/v2c_part2/output_origin
)

python V2C/merge_and_wer.py \
    --src "${src[@]}" \
    --dst /data2/ruixin/datasets/v2c_raw_all