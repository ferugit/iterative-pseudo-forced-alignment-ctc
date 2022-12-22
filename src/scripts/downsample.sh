#!/bin/bash

src_audio_path=/disk3/data/fernandol/data/Albayzin/RTVE2020DB/RTVE2020DB/test/audio/S2T
dst_audio_path=/disk3/data/fernandol/data/Albayzin/RTVE2020DB/RTVE2020DB/test/audio_16kHz/S2T

sample_rate=16000

for file in $src_audio_path/*.aac;
do
filename="$(basename $file)"
ffmpeg -i ${file} -ar $sample_rate $dst_audio_path/${filename/.aac/.wav} -y; 
done