import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse

import torchaudio

import pandas as pd

import utils.text_utils as text_utils


def main(args):

    print('Scanning train split.')
    filename = os.path.join(args.dst, 'train.tsv')

    if(os.path.isfile(filename)):
        print(filename + ' file already exists, so it is not created.')
    
    else:
        train_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'train/audio/')):

            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):
                    stm_file_name = os.path.join(args.src, 'train/stm/', file).replace('aac', 'stm')

                    with open(stm_file_name, 'r') as stm_file:
                        
                        for line_stm in stm_file.readlines():
                            stm_data = line_stm.strip().split(' ', 6)
                            
                            if(len(stm_data) < 7):
                                print('Bad format in line: ' + str(line_stm))
                                print('Skiping to next line...')
                                continue

                            speaker_id = stm_data[2]
                            start = float(stm_data[3])
                            end = float(stm_data[4])
                            audio_length = end - start
                            channel = stm_data[1]
                            sample_id = stm_data[0] + '_{0}_{1}'.format(start, end)
                            sample_path = os.path.join(root, file)
                            sample_path = sample_path.replace(
                                'train/audio/', 
                                'train/audio_16kHz/'
                                ).replace(
                                    '.aac',
                                    '.wav'
                                )
                                
                            transcription = text_utils.normalize_transcript(stm_data[6])
                            train_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2022'])

        train_df = pd.DataFrame(train_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        train_df.to_csv(filename, sep='\t', index=None)


    print('Scanning test split.')
    filename = os.path.join(args.dst, 'test.tsv')

    if(os.path.isfile(filename)):
        print(filename + ' file already exists, so it is not created.')
    
    else:
        test_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'test/audio/S2T/')):

            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):
                    sample_path = os.path.join(root, file)
                    sample_path = sample_path.replace(
                        'test/audio/', 
                        'test/audio_16kHz/'
                        ).replace(
                            '.aac',
                            '.wav'
                        )
                    
                    # audio length
                    audio_metadata = torchaudio.info(sample_path)
                    full_audio_length = audio_metadata.num_frames/audio_metadata.sample_rate


                    speaker_id = "unknown"
                    start = 0.0
                    end = full_audio_length
                    audio_length = end - start
                    channel = "unknown"
                    sample_id = file.replace(".aac", ".wav") + '_{0}_{1}'.format(start, end)
                    transcription = "unknown"
                    test_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2022'])

        test_df = pd.DataFrame(test_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        test_df.to_csv(filename, sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate tsv from RTVE2022DB')
    parser.add_argument('--src', help='source directory of metadata', default='/disks/md1-8T/audio/Albayzin/RTVE2020DB/')
    parser.add_argument('--dst', help='destination directory of metadata', default='data/RTVE2020/tsv/')
    args = parser.parse_args()

    main(args)