export CUDA_VISIBLE_DEVICES=1

python build_movie_origin.py \
    --movie-map V2C/DataConstruction/movie_video_map.csv \
    --utterances V2C/DataConstruction/movie_speaker_emotion.csv \
    --video-root /data2/ruixin/downloads/v2c_anim \
    --output-root /data2/ruixin/datasets/v2c_origin