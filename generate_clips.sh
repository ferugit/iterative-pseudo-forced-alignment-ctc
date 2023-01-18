#!/bin/bash
#
# Created by Fernando López Gavilánez (2023)
# 
# This script generates clips from aligned data
# Used columns: Sample_Path, Sample_ID, Start, End

src=data/wip_benedetti_words/results/benedetti_aligned_words.tsv
dst=data/wip_benedetti_words/clips/
mkdir -p $dst

python -u src/scripts/generate_clips.py --src $src --dst $dst