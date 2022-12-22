import sys
sys.path.insert(1, sys.path[0].replace('/test', ''))

import pandas as pd
import utils.text_utils as text_utils


def prepare_text(transcript, max_words_sequence=24):
    """
    Prepare text for alignment.
    If text is longer than max_word_sequence, then is splited.
    
    Arguments
    ---------
    transcript: new text to align
    max_words_sequence: maximum word length
    
    Returns
    -------
    transcript: list of text to align
    """
    if(len(transcript.split(' ')) > max_words_sequence):
        transcript = text_utils.split_long_transcript(
            transcript,
            max_words_sequence=max_words_sequence
            )
    else:
        transcript = [transcript] # transform to list
    # NOTE: Using the space between utterances provides worse results
    # list_to_align = []
    #for i in range(1,  2*len(transcript)):
        #list_to_align.append("·" if(i+1)%2 else transcript[int(i/2)])
        #transcript = list_to_align
    return transcript

file_path = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2018/segmented/20H-20171211.tsv'
original_path = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2018/tsv/dev1.tsv'

aligned_df = pd.read_csv(file_path, header=0, sep='\t')
original_df = pd.read_csv(original_path, header=0, sep='\t')

original_df = original_df[original_df['Sample_Path'] == aligned_df.sample(1)['Sample_Path'].values[0]]
original_df.reset_index(drop=True, inplace=True)

aligned_text = aligned_df['Transcription'].tolist()
original_text = aligned_df['Transcription'].tolist()
processed_text = []

for text in original_text:
    processed_text += prepare_text(text)

processed_text = [x.upper() for x in processed_text]

for index in range(len(aligned_text)):
    print(aligned_text[index])
    print(processed_text[index])
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

    if aligned_text[index] != processed_text[index]:
        print(
            "At index {0} text aligned is {1} and it should be {2}".format(
                index,
                aligned_text[index],
                processed_text[index]
            )
        )
        break