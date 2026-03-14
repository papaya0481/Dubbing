ost=/data2/ruixin/datasets/flow_dataset/MELD/semantic/audios/ost
dst=/data2/ruixin/datasets/flow_dataset/MELD/semantic/audios/aligned

mfa align "$ost" \
    english_us_arpa english_us_arpa \
    "$dst" \
    --clean --single_speaker
