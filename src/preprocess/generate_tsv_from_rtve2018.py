import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse

import pysrt
import pandas as pd

import torchaudio

import utils.text_utils as text_utils
from difflib import SequenceMatcher


def main(args):

    print('Scanning RTVE2018 Database.')

    # Scan RTVE2018 train files: closed-captions (srt)
    print('Scanning train split.')
    
    if(os.path.isfile(os.path.join(args.dst, 'train.tsv'))):
        print(os.path.join(args.dst, 'train.tsv') + ' file already exists, so it is not created.')
    
    else:
        train_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'train/audio')):
            
            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):

                    try:
                        # Open respective subtitles file
                        subtitles =  pysrt.open(os.path.join(args.src, 'train/srt', file.replace('aac', 'srt')), encoding='utf-8')

                    except Exception as e:
                        print('File ' + str(os.path.join(args.src, 'train/srt', file.replace('aac', 'srt'))) + ' is not encoded in UTF-8! Using Latin-1.')
                        subtitles =  pysrt.open(os.path.join(args.src, 'train/srt', file.replace('aac', 'srt')), encoding='latin-1')
                    
                    # Remove time offset: hours, minutes and seconds (?)
                    fisrt_subtitle_start = subtitles[0].start.to_time()
                    start_offset = fisrt_subtitle_start.hour*3600 + fisrt_subtitle_start.minute*60

                    # Audio path: use downsampled audio
                    sample_path = os.path.join(root, file)
                    sample_path = sample_path.replace(
                    'train/audio', 
                    'train/audio_16kHz').replace(
                        '.aac',
                        '.wav'
                    )

                    # Get audio length
                    audio_metadata = torchaudio.info(sample_path)
                    full_audio_length = audio_metadata.num_frames/audio_metadata.sample_rate

                    # For every subtitle in audio file
                    for sub in subtitles:
                        
                        start = sub.start.to_time()
                        start = start.hour*3600 + start.minute*60 + start.second + start.microsecond*0.000001 - start_offset
                        end = sub.end.to_time()
                        end = end.hour*3600 + end.minute*60 + end.second + end.microsecond*0.000001 -start_offset
                        audio_length = end - start
                        sample_id = file.replace('.acc', '') + '_{0}_{1}'.format(start, end)
                        # Annotations
                        if '(' == sub.text[0] and ')' == sub.text[-1]:
                            print(sub.text)
                            continue
                        transcription = text_utils.normalize_transcript(sub.text)
                        train_list.append([sample_id, sample_path, audio_length, start, end, transcription, "Unknown", 'RTVE2018'])
                        
        rtve2018_train_df = pd.DataFrame(train_list, columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        rtve2018_train_df['Channel'] = 1
        rtve2018_train_df.to_csv(os.path.join(args.dst, 'train.tsv'), sep='\t', index=None)
    
    # dev1 data: closed-captions and human revised labels (stm and trn)
    print('Scanning dev1 split.')
    filename = os.path.join(args.dst, 'dev1.tsv')

    if(os.path.isfile(filename)):
        print(filename + ' file already exists, so it is not created.')
    
    else:
        dev1_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'dev1/audio')):

            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):            
                    stm_file_name = os.path.join(args.src, 'dev1/stm', file).replace('aac', 'stm')
                    trn_file_name = os.path.join(args.src, 'dev1/trn', file).replace('aac', 'trn')

                    with open(stm_file_name, 'r') as stm_file, open(trn_file_name) as trn_file:
                        continue_to_next_line = False
                        
                        for line_stm in stm_file.readlines():

                            stm_data = line_stm.strip().split(' ', 6)
                            
                            if(len(stm_data) < 7):
                                print('Bad format in line: ' + str(line_stm))
                                print('Skiping to next line...')
                                continue_to_next_line = True
                            
                            speaker_id = stm_data[2]
                            
                            while True:
                                trn_data = trn_file.readline().strip().split(' ', 1)

                                if(len(trn_data) < 2):
                                    print('Bad format in line: ' + str(trn_data))
                                    print('Skiping to next line...')
                                    continue_to_next_line = True
                                    break
                                
                                if(trn_data[0] == '(#_1000'):
                                    continue
                                elif(trn_data[0].replace('(', '').replace(')', '') == speaker_id):
                                    break
                            
                            if(continue_to_next_line):
                                continue_to_next_line = False
                                continue

                            start = float(stm_data[3])
                            end = float(stm_data[4])
                            audio_length = end - start
                            channel = stm_data[1]
                            sample_id = stm_data[0] + '_{0}_{1}'.format(start, end)
                            sample_path = os.path.join(root, file)
                            sample_path = sample_path.replace(
                                'dev1/audio', 
                                'dev1/audio_16kHz').replace(
                                    '.aac',
                                    '.wav'
                                )
                            transcription = text_utils.normalize_transcript(trn_data[1])

                            if(SequenceMatcher(None, transcription, stm_data[6]).ratio() < 0.8):
                                print('\nToo much difference between transcriptions: ')
                                print('\t' + transcription)
                                print('\t' + stm_data[6])

                            dev1_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2018'])
                    
        rtve2018_dev1_df = pd.DataFrame(dev1_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        rtve2018_dev1_df.to_csv(filename, sep='\t', index=None)
    

    # dev2 data: closed-captions and human revised labels (stm and trn)
    print('Scanning dev2 split.')
    filename = os.path.join(args.dst, 'dev2.tsv')

    if(os.path.isfile(filename)):
        print(filename + ' file already exists, so it is not created.')
    
    else:
        dev2_list = []

        for root, _, files in os.walk(os.path.join(args.src, 'dev2/audio')):

            for file in files:
                
                if file.endswith('.aac') and not file.startswith('.'):
                    
                    stm_file_name = os.path.join(args.src, 'dev2/stm', file).replace('aac', 'stm')
                    trn_file_name = os.path.join(args.src, 'dev2/trn', file).replace('aac', 'trn')

                    with open(stm_file_name, 'r') as stm_file, open(trn_file_name) as trn_file:

                        continue_to_next_line = False
                        
                        for line_stm in stm_file.readlines():

                            stm_data = line_stm.strip().split(' ', 6)
                            
                            if(len(stm_data) < 7):
                                print('Bad format in line: ' + str(line_stm))
                                print('Skiping to next line...')
                                continue_to_next_line = True
                            
                            speaker_id = stm_data[2]
                            
                            while True:
                                trn_data = trn_file.readline().strip().split(' ', 1)

                                if(len(trn_data) < 2):
                                    print('Bad format in line: ' + str(trn_data))
                                    print('Skiping to next line...')
                                    continue_to_next_line = True
                                    break
                                
                                if(trn_data[0] == '(#_1000'):
                                    continue
                                elif(trn_data[0].replace('(', '').replace(')', '') == speaker_id):
                                    break
                            
                            if(continue_to_next_line):
                                continue_to_next_line = False
                                continue

                            start = float(stm_data[3])
                            end = float(stm_data[4])
                            audio_length = end - start
                            channel = stm_data[1]
                            sample_id = stm_data[0] + '_{0}_{1}'.format(start, end)
                            sample_path = os.path.join(root, file)
                            sample_path = sample_path.replace(
                                'dev2/audio', 
                                'dev2/audio_16kHz').replace(
                                    '.aac',
                                    '.wav'
                                )
                            transcription = text_utils.normalize_transcript(trn_data[1])

                            if(SequenceMatcher(None, transcription, stm_data[6]).ratio() < 0.8):
                                print('\nToo much differences between transcriptions: ')
                                print('\t' + transcription)
                                print('\t' + stm_data[6])

                            dev2_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2018'])
                    
        rtve2018_dev2_df = pd.DataFrame(dev2_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        rtve2018_dev2_df.to_csv(filename, sep='\t', index=None)

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
                    stm_file_name = os.path.join(args.src, 'test/references/stm', file).replace('aac', 'stm')
                    if 'LM-20170103' in stm_file_name:
                        stm_file_name = stm_file_name.replace('20170103', '20170103-MD')

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
                                'test/audio/S2T/', 
                                'test/audio_16kHz/S2T/').replace(
                                    '.aac',
                                    '.wav'
                                )
                            transcription = text_utils.normalize_transcript(stm_data[6])
                            test_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, 'RTVE2018'])

        rtve2018_test_df = pd.DataFrame(test_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        rtve2018_test_df.to_csv(filename, sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to search wanted words in Albayzin database')
    parser.add_argument('--src', help='source directory of metadata', default='/disks/md1-8T/audio/Albayzin/RTVE2018DB/')
    parser.add_argument('--dst', help='destination directory of metadata', default='data/RTVE2018/tsv/')
    args = parser.parse_args()

    main(args)