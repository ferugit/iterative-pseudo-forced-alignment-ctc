#!/bin/bash
tsv=/disks/md1-8T/users/cx02275/data/Albayzin-aligned/tsv/rs_albayzin_aligned.tsv

python -u src/scripts/generate_hypothesis_reference.py --tsv $tsv