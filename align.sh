#!/bin/bash
#
# Created by Fernando López Gavilánez (2022)
#
# This script generates alignmnets of long audio files using the method proposed in https://arxiv.org/abs/2210.15226 
#
# Requirements:
#   i) Audio files and a tsv file containing the following columns: 
#       [Sample_ID, Sample_Path, Channel, Audio_Length, Start, End, Transcription, Speaker_ID, Database]
#       
#       *Sample_ID*: audio filename w/o extension + "_" + START_OF_SEGMENT + "_" + END_OF_SEGMENT
#       *Sample_Path*: audio file path
#       *Channel*: channel
#       *Audio_Length*: END_OF_SEGMENT - START_OF_SEGMENT
#       *Start*: START_OF_SEGMENT
#       *End*: END_OF_SEGMENT
#       *Transcription*: reference text
#       *Speaker_ID*: not used
#       *Database*: not used
#       
#   ii) A already trained ASR in the target language in the SpeechBrain language
#
# Process:
#   i) Generate VAD segments
#   ii) Filter VAD segments
#   iii) Iterative alignment


#########################################################
###################### DEFINITIONS ######################
#########################################################


# config zone
alignment_name="rtve2022_dev" # alignment name, comment to use timestamp instead
tsv_path=data/dev/tsv/dev.tsv # source file with metadata
generate_vad_segments=false # put to false if already generated
generate_stm_results=true # generate stm files from tsv results
n_process=1 # number of processes to perform alignment, numbers bigger than 1 perform parallel alignment

# VAD configuration
max_non_speech_segments=20.0 # vad segments to filter
# add more parameters

# Alignment parameters
threshold=-2.0 # anchors threshold
short_utterance_len=30 # Minimum sequence of chars to select anchors
min_words_sequence=NULL # Not used for the moment
max_words_sequence=100 # Mesured from CommonVoice Dataset
max_window_size=70.0 # Seconds, for the sake of computational load
window_to_stop=500.0 # Seconds, windows to stop execution
min_text_to_audio_prop=0.8 # Min text to audio proportion
max_text_to_audio_prop_exec=10 # Number of consecutive exceptions to stop

# trained ASR
asr_src_path="data/asr/"
asr_yaml="ctc_sp_with_wav2vec.yaml"
asr_savedir="data/savedir"

#########################################################
####################### ALIGNMENT #######################
#########################################################


# get tsv filename
readarray -d / -t strarr <<<"$tsv_path"
tsv_filename=${strarr[${#strarr[@]} - 1]}

# generate WIP directories
if [ ! -z ${alignment_name+set} ];
then
    wip_dir="data/wip_"$alignment_name
    echo "Alignment name defined, WIP folder is: "$wip_dir
else
    wip_dir="data/wip_"$(date +%s)
    echo "Alignment name not defined, WIP folder is: "$wip_dir
fi

vad_dir=$wip_dir"/vad"
results_dir=$wip_dir"/results"
logs_dir=$wip_dir"/logs"

mkdir -p $wip_dir
mkdir -p $vad_dir
mkdir -p $results_dir
mkdir -p $logs_dir

# get vad segments
if $generate_vad_segments
then    
    echo "Generating VAD segments: "$tsv_path
    python -u src/preprocess/get_vad_segments_speechbrain.py --src $tsv_path --dst $vad_dir
fi

vad_segments_tsv=${tsv_filename/.tsv/_vad_segments.tsv}
vad_segments_filtered_tsv=${vad_segments_tsv/.tsv/_filtered.tsv}
vad_segments_filepath=$vad_dir"/"$vad_segments_tsv
vad_segments_filtered_filepath=$vad_dir"/"$vad_segments_filtered_tsv

# filter vad segments
python -u src/preprocess/filter_non_speech_segments.py --src $vad_segments_filepath --dst $vad_dir --length $max_non_speech_segments

# get alignment
for (( i=0; i<$n_process; i++ ))
do
    python -u src/iterative_utterance_alignment.py --tsv $tsv_path --vad_segments_tsv $vad_segments_filtered_filepath \
    --dst $results_dir --asr_src_path $asr_src_path --asr_yaml $asr_yaml --asr_savedir $asr_savedir --threshold $threshold \
    --logs_path $logs_dir --short_utterance_len $short_utterance_len --max_words_sequence $max_words_sequence \
    --max_window_size $max_window_size --window_to_stop $window_to_stop --min_text_to_audio_prop $min_text_to_audio_prop \
    --max_text_to_audio_prop_exec $max_text_to_audio_prop_exec > $logs_dir"/global_"${i}.log &
done

# Wait for the processes to finish
wait

# generate stm files
if $generate_stm_results
then
    stm_dir=$results_dir/stm
    mkdir -p $stm_dir
    python -u src/scripts/tsv_to_stm.py --src_path $results_dir --dst_path $stm_dir
fi