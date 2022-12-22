#!/bin/bash

dst_path=data/alignment_analysis/

# train rtve2018
name=train_aligned.tsv
tsv=data/RTVE2018DB/segmented/3rd_pass/$name
python -u src/scripts/generate_transcription_and_wer.py --tsv $tsv --dst_path $dst_path --name $name

# dev1 rtve2018
name=dev1_aligned.tsv
tsv=data/RTVE2018DB/segmented/3rd_pass/$name
python -u src/scripts/generate_transcription_and_wer.py --tsv $tsv --dst_path $dst_path --name $name

# dev2 rtve2018
name=dev2_aligned.tsv
tsv=data/RTVE2018DB/segmented/3rd_pass/$name
python -u src/scripts/generate_transcription_and_wer.py --tsv $tsv --dst_path $dst_path --name $name

# test rtve2018
name=test_aligned.tsv
tsv=data/RTVE2018DB/segmented/3rd_pass/$name
python -u src/scripts/generate_transcription_and_wer.py --tsv $tsv --dst_path $dst_path --name $name

# train rtve2022
name=train22_aligned.tsv
tsv=data/RTVE2022DB/segmented/3rd_pass/train_aligned.tsv
python -u src/scripts/generate_transcription_and_wer.py --tsv $tsv --dst_path $dst_path --name $name