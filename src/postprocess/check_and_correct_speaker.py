import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse

import pandas as pd

from utils import text_utils


def fix_aligned_text(df_1, reference_df):
    """
    This removes duplicated  index.
    """
    repeated_index = []
    
    df_len = len(reference_df.index)
    df_iterator = reference_df.iterrows()
    ref_idx, ref_row = next(df_iterator)
    ref_text = ref_row['Transcription']

    for idx, row in df_1.iterrows():
        text = row['Transcription']
        
        if text != ref_text:
            repeated_index.append(idx)
            continue
        
        if ref_idx + 1 < df_len:
            ref_idx, ref_row = next(df_iterator)
            ref_text = ref_row['Transcription']
        else:
            break

    return df_1[~df_1.index.isin(repeated_index)].reset_index()


def same_diff(a, b):
    """Align two list with misses and inclusions.
    """
    sames = []
    diffs = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            sames.append(a[i])
            i += 1
            j += 1
        elif a[i] > b[j]:
            diffs.append(b[j])
            j += 1
        else:
            diffs.append(a[i])
            i += 1
    diffs += a[i:]
    diffs += b[j:]
    return sames, diffs


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


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
    return transcript


def main(args):
    
    # Check if paths exist
    if(os.path.isfile(args.partition) and os.path.isdir(args.src)):
        
        # Read partition information
        partition_df = pd.read_csv(args.partition, header=0, sep='\t')
        audio_paths = partition_df['Sample_Path'].unique()
        destination_path = os.path.join(args.src, 'reviewed')
        errors_path = os.path.join(destination_path, 'errors')
        check_path(destination_path)
        check_path(errors_path)

        # Iterate over all audio paths
        for audio_path in audio_paths:
            aligned_filename = os.path.join(args.src, audio_path.split('/')[-1].replace('.wav', '.tsv'))

            #if not 'millennium-20170522.tsv' in aligned_filename:
            #    continue

            # If the file has been aligned
            if(os.path.isfile(aligned_filename)):
                aligned_df = pd.read_csv(aligned_filename, header=0, sep='\t')
                filename = aligned_filename.split('/')[-1]
                result_path = os.path.join(destination_path, filename)

                # If result already exist do not repeat it
                if(os.path.isfile(result_path)):
                    continue
                
                reference_df = partition_df[partition_df['Sample_Path'] == audio_path]
                reference_list = []

                # Get correct references
                for idx, row in reference_df.iterrows():
                    transcript = row['Transcription'].upper()
                    speaker_id = row['Speaker_ID']
                    transcript = prepare_text(transcript)
                    
                    for utterance in transcript:
                        reference_list.append([utterance, speaker_id])

                new_reference_df = pd.DataFrame(reference_list, columns=['Transcription', 'Speaker_ID'])
                
                # Check if references match with aligned file
                if(aligned_df['Transcription'].equals(new_reference_df['Transcription'])):
                    aligned_df['Transcription'] = new_reference_df['Transcription']
                    aligned_df['Speaker_ID'] = new_reference_df['Speaker_ID']
                    aligned_df.to_csv(result_path, index=None, sep='\t')
                    print('File {0} has been reviewed. It has correct transcription and Speaker ID.'.format(filename))

                # If not matching try to correct it: in the future this will not be needed.
                else:
                    print('File {0} has been reviewed. Not matching transcription. Trying to correct...'.format(filename))
                    aligned_list = aligned_df['Transcription'].tolist()
                    new_ref_list = new_reference_df['Transcription'].tolist()

                    # Aligned dataframe has duplicates
                    if len(new_ref_list) < len(aligned_list):
                        aligned_df = fix_aligned_text(aligned_df, new_reference_df)
                    
                    # If something weird happend
                    else:
                        same, _= same_diff(new_ref_list, aligned_list)
                        # TODO: keep working here does not make much sense, as this should be fixed from source
                        #aligned_df = aligned_df[aligned_df['Transcription'].isin(same)].reset_index()
                        #new_reference_df = new_reference_df[new_reference_df['Transcription'].isin(same)].reset_index()
                    
                    aligned_list = aligned_df['Transcription'].tolist()
                    new_ref_list = new_reference_df['Transcription'].tolist()

                    if(aligned_df['Transcription'].equals(new_reference_df['Transcription'])):
                        aligned_df['Transcription'] = new_reference_df['Transcription']
                        aligned_df['Speaker_ID'] = new_reference_df['Speaker_ID']
                        aligned_df.to_csv(result_path, index=None, sep='\t')
                        print('Successfully corrected.')
                    else:
                        print('The solution has not been found, copying the original file by the moment.')
                        aligned_df.to_csv(result_path, index=None, sep='\t')
                        new_reference_df['Aligned_Text'] = aligned_df['Transcription']
                        new_reference_df.to_csv(os.path.join(errors_path, filename), index=None, sep='\t')

    else:
        print('Partition file or source directory does not exist')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script correct RTVE2018DB speaker after alignment")
    parser.add_argument("--partition", help="partition that has been aligned", default="")
    parser.add_argument("--src", help="folder with aligned files", default="")
    args = parser.parse_args()

    # Run main
    main(args)