#!/bin/bash
#
# Created by Fernando López Gavilánez (2023)
#
# This script generates alignmnets of long audio files using the method proposed in https://arxiv.org/abs/2210.15226 
#
# Requirements:
#   i) Audio files and a tsv file containing the following columns: 
#       [Sample_ID, Sample_Path, Channel, Audio_Length, Start, End, Transcription, Speaker_ID, Database]
#       
#       *** The source temporal information is not used, can be dummy temporal information.
#       
#       +---------------+-----+--------------------------------------------------------+
#       |     Name      | Use |                      Explanation                       |
#       +---------------+-----+--------------------------------------------------------+
#       | Sample_ID     | Yes | Unique sample identifier (e.g. AG-20210605_9.04_14.08) |
#       | Sample_Path   | Yes | Audio file path (e.g. /path/to/file/AG-20210605.wav)   |
#       | Channel       | No  | e.g. 1                                                 |
#       | Audio_Length  | No  | END_OF_SEGMENT - START_OF_SEGMENT                      |
#       | Start         | No  | START_OF_SEGMENT                                       |
#       | End           | No  | END_OF_SEGMENT                                         |
#       | Transcription | Yes | Text to be aligned                                     |
#       | Speaker_ID    | No  | Speaker identifier                                     |
#       | Database      | No  | Database identifier                                    |
#       +---------------+-----+--------------------------------------------------------+
#       
#   ii) An already trained ASR in the target language in the SpeechBrain framework (EncoderASR)
#
# Process:
#   i) Generate VAD segments (Optional)
#   ii) Filter VAD segments
#   iii) Iterative alignment
#   iv) Merge aligned files (Optional)
#   v) Generate stm resultant files (Optional)
#
#
# Results:
#   i) VAD segmentation (Optional)
#   ii) VAD segmentation filtered
#   iii) A tsv file for every aligned audio file
#   iv) A tsv file with all the alignments (Optional)
#   v) stm files of the aligned data (Optional)
#


#########################################################
###################### DEFINITIONS ######################
#########################################################


# config zone
#alignment_name="rtve2022_dev" # alignment name, comment to use timestamp instead
#tsv_path=data/dev/tsv/dev.tsv # source file with metadata
alignment_name="benedetti" # alignment name, comment to use timestamp instead
tsv_path=data/sample/tsv/benedetti.tsv # source file with metadata
merge_files=true # merge aligned files in a single tsv
generate_vad_segments=true # put to false if already generated
generate_stm_results=true # generate stm files from tsv results
n_process=1 # number of processes to perform alignment, numbers bigger than 1 perform parallel alignment

# VAD configuration
max_non_speech_segments=20.0 # vad segments to filter
# TODO: add more parameters

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

# remove empty files from previous executions
echo "Removing previous empty files from: "$results_dir
find $results_dir -type f -empty -print -delete

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
echo "Filtering VAD segments..."
python -u src/preprocess/filter_non_speech_segments.py --src $vad_segments_filepath --dst $vad_dir --length $max_non_speech_segments

# get alignment
echo "Starting alignment..."
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

# merge results in a single file
if $merge_files
then
    echo "Merging aligned files from: "$results_dir
    python -u src/postprocess/merge_aligned_files.py --global_tsv $tsv_path --src $results_dir
fi

# generate stm files
if $generate_stm_results
then
    echo "Generating stm files from: "$results_dir
    stm_dir=$results_dir/stm
    mkdir -p $stm_dir
    python -u src/scripts/tsv_to_stm.py --src_path $results_dir --dst_path $stm_dir
fi
