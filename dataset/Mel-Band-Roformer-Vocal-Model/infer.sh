# dataset="MELD_clips"
# folder="dev"

# python inference.py \
#     --config_path configs/config_vocals_mel_band_roformer.yaml \
#     --model_path /data2/ruixin/downloads/melbandroformer/MelBandRoformer.ckpt \
#     --input_folder /data2/ruixin/datasets/$dataset/audios/ost/$folder \
#     --store_dir /data2/ruixin/datasets/$dataset/audios/vocals/$folder


dataset="v2c_clips"
folder=(
    "Brave" "Cloudy" "CloudyII" "CoCo"
)

for f in "${folder[@]}"; do
    python inference.py \
        --config_path configs/config_vocals_mel_band_roformer.yaml \
        --model_path /data2/ruixin/downloads/melbandroformer/MelBandRoformer.ckpt \
        --input_folder /data2/ruixin/datasets/$dataset/audios/ost/$f \
        --store_dir /data2/ruixin/datasets/$dataset/audios/vocals/$f
done