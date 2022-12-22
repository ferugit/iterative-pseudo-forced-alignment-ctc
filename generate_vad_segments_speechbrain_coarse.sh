#!/bin/bash
src=data/RTVE2018DB/tsv/
dst=data/RTVE2018DB/vad_segments/
splits=(dev1.tsv dev2.tsv test.tsv)

for split in ${splits[@]}
do
    echo "Processing file: "$src$split
    python -u src/preprocess/get_vad_segments_speechbrain.py --src $src$split --dst $dst
    python -u src/preprocess/filter_non_speech_segments.py --src $dst$split --dst $dst
done