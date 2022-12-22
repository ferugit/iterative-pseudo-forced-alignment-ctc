#!/bin/bash

# train split RTVE2018
src=data/RTVE2018DB/segmented/3rd_pass/train_aligned_filtered.tsv
dst=data/RTVE2018DB/segmented_clips/train/

python -u src/scripts/generate_clips.py --src $src --dst $dst

# dev1 split RTVE2018
src=data/RTVE2018DB/segmented/3rd_pass/dev1_aligned_filtered.tsv
dst=data/RTVE2018DB/segmented_clips/dev1/

python -u src/scripts/generate_clips.py --src $src --dst $dst

# dev2 split RTVE2018
src=data/RTVE2018DB/segmented/3rd_pass/dev2_aligned_filtered.tsv
dst=data/RTVE2018DB/segmented_clips/dev2/

python -u src/scripts/generate_clips.py --src $src --dst $dst


# test split RTVE2018
src=data/RTVE2018DB/segmented/3rd_pass/test_aligned_filtered.tsv
dst=data/RTVE2022DB/segmented_clips/test/

python -u src/scripts/generate_clips.py --src $src --dst $dst

# train split RTVE2022
src=data/RTVE2022DB/segmented/3rd_pass/train_aligned_filtered.tsv
dst=data/RTVE2022DB/segmented_clips/train/

python -u src/scripts/generate_clips.py --src $src --dst $dst