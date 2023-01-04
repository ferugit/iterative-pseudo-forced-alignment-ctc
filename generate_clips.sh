#!/bin/bash
#
# Created by Fernando López Gavilánez (2023)
# 
# This script generates clips from aligned data
# Used columns: Sample_Path, Sample_ID, Start, End

#src=data/wip_benedetti/results/benedetti_aligned_filtered.tsv
#dst=data/wip_benedetti/clips/

src=data/wip_benedetti_words/results/filtered_aligned.tsv
dst=data/wip_benedetti_words/clips

python -u src/scripts/generate_clips.py --src $src --dst $dst