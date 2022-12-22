#!/bin/bash

data=data/RTVE2018DB/
dst=data/RTVE2018DB/segmented/

n_process=5
for (( i=0; i<$n_process; i++ ))
do
    python -u src/iterative_utterance_alignment.py --tsv $data --dst $dst > ${i}.log &
done

# Wait for the processes to finish
wait

data=data/RTVE2022DB/
dst=data/RTVE2022DB/segmented/

n_process=5
for (( i=0; i<$n_process; i++ ))
do
    python -u src/iterative_utterance_alignment.py --tsv $data --dst $dst > ${i}.log &
done

# Wait for the processes to finish
wait