#!/bin/bash
src=/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2018/tsv/
dst=/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2018/vad_segments
splits=(dev1.tsv dev2.tsv train.tsv)

for split in ${splits[@]}
do
    echo "Processing file: "$src$split
    python -u src/scripts/get_vad_segments_rvad.py --src $src$split --dst $dst
done