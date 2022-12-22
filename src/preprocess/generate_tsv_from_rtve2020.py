import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse

import pandas as pd

import utils.text_utils as text_utils


def main(args):

# test data: closed-captions and human revised labels (stm and trn)
    print('Scanning test split.')
    filename = os.path.join(args.dst, 'test.tsv')

    if(os.path.isfile(filename)):
        print(filename + ' file already exists, so it is not created.')
    
    else:
        test_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'test/audio/S2T/')):

            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):
                    
                    # stm files do not have low bar in name
                    if '_' in file:
                        stm_file_name = os.path.join(args.src, 'test/references/S2T/stm/', file.replace('_', '-')).replace('aac', 'stm')
                    else:
                        stm_file_name = os.path.join(args.src, 'test/references/S2T/stm/', file).replace('aac', 'stm')
                    
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
                                'test/audio/', 
                                'test/audio_16kHz/'
                                ).replace(
                                    '.aac',
                                    '.wav'
                                )
                                
                            transcription = text_utils.normalize_transcript(stm_data[6])
                            test_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2018'])

        rtve2018_test_df = pd.DataFrame(test_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        rtve2018_test_df.to_csv(filename, sep='\t', index=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to search wanted words in Albayzin database')
    parser.add_argument('--src', help='source directory of metadata', default='/disks/md1-8T/audio/Albayzin/RTVE2020DB/')
    parser.add_argument('--dst', help='destination directory of metadata', default='data/RTVE2020/tsv/')
    args = parser.parse_args()

    main(args)