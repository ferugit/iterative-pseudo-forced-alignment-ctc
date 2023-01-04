#!/bin/bash
#
# Created by Fernando López Gavilánez (2023)
#
# This script generates sub-uttrance alignmnets using the method proposed in https://arxiv.org/abs/2210.15226 
#
# Requirements:
#   i) Audio files and a tsv file containing the following columns: 
#       [Sample_ID, Sample_Path, Channel, Audio_Length, Start, End, Segment_Score, Transcription, Speaker_ID, Database] + others
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
#   i) Search wanted text in transcriptions
#   ii) Perform word-level alignment
#
# Results:
#   i) A tsv with the filtered data: data that contains the word in text reference
#   ii) A tsv with thow wanted words aligned
#


#########################################################
###################### DEFINITIONS ######################
#########################################################

# config zone
config_file=config/words.json # json config file: contains an array with the wanted words
alignment_name="benedetti_words" # alignment name, comment to use timestamp instead
tsv_path=data/wip_benedetti/results/benedetti_aligned.tsv # source file with metadata
text_column="Transcription" # column name in tsv that contains the utterance text reference

# alignment corrections: better apply this after
collar=0.0 # collar to alignment in seconds
offset_time=0.0 # alignment shift to rigth in seconds
left_offset=0.0 # start shift in seconds
right_offset=0.0 # end shift in seconds

# trained ASR
asr_src_path="data/asr/"
asr_yaml="ctc_sp_with_wav2vec.yaml"
asr_savedir="data/savedir"


#########################################################
####################### ALIGNMENT #######################
#########################################################


# generate WIP directories
if [ ! -z ${alignment_name+set} ];
then
    wip_dir="data/wip_"$alignment_name
    echo "Alignment name defined, WIP folder is: "$wip_dir
else
    wip_dir="data/wip_"$(date +%s)
    echo "Alignment name not defined, WIP folder is: "$wip_dir
fi

results_dir=$wip_dir"/results"
logs_dir=$wip_dir"/logs"

mkdir -p $wip_dir
mkdir -p $results_dir
mkdir -p $logs_dir

# get tsv filename
readarray -d / -t strarr <<<"$tsv_path"
tsv_filename=${strarr[${#strarr[@]} - 1]}

# destination tsv
filtered_tsv_dir=$results_dir"/"${tsv_filename/.tsv/_filtered.tsv}

# filter utterances where wanted word appears
echo "Searching words in source data..."
python -u src/search_words.py --tsv_path $tsv_path --dst $results_dir \
 --config_file $config_file --text_column $text_column

# perform alignment
echo "Starting word-level alignment..."
python -u src/word_level_alignment.py --tsv $filtered_tsv_dir \
 --dst_path $results_dir --asr_src_path $asr_src_path --asr_yaml $asr_yaml \
 --asr_savedir $asr_savedir --logs_path $logs_dir