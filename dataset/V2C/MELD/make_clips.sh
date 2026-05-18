export CUDA_VISIBLE_DEVICES=2

python extract_clips.py \
    --episode-map /data2/ruixin/downloads/friends/friends_episode_map.csv \
    --output-dir /data2/ruixin/datasets/MELD_clips \
    --map-root /data2/ruixin/downloads/friends/ \
    --asr-check