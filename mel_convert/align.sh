ost=/data2/ruixin/datasets/flow_dataset/MELD_semantic/audios/ost
dst=/data2/ruixin/datasets/flow_dataset/MELD_semantic/audios/aligned

mfa align "$ost" \
    english_us_arpa english_us_arpa \
    "$dst" \
    --clean --single_speaker
