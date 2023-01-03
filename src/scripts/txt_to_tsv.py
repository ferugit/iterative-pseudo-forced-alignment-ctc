import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import argparse
import pandas as pd

import utils.text_utils as text_utils
import utils.time_utils as time_utils


def main(args):

    working_dir = sys.path[0].replace(sys.path[0].split('/')[-1], '').replace(sys.path[0].split('/')[-2], '')
    src_path = os.path.join(working_dir, args.src_path)
    dst_path = os.path.join(working_dir, args.dst_path)

    if (not os.path.isdir(src_path)) and (not os.path.isdir(dst_path)) and args.database_name:
        print("Non valid arguments to convert txt files to tsv.")

    else:
        tsv_list = []
        tsv_file_name = os.path.join(dst_path, args.database_name + '.tsv')

        for root, _, files in os.walk(src_path):

            for file in files:
                    
                    if file.endswith('.txt') and not file.startswith('.'):
                        audio_file = file.replace('.txt', '.wav')

                        with open(os.path.join(src_path, file), 'r') as src_file:
                            lines = src_file.readlines()
                            list_len = len(lines)

                            for idx in range(list_len//2):
                                cursor = idx*2
                                first_line = lines[cursor]
                                second_line = lines[cursor+1]

                                if idx < ((list_len//2) -1):
                                    thrid_line = lines[cursor+2]
                                else:
                                    thrid_line = first_line # wrong info
                                
                                start = float(time_utils.get_sec_h_m_s(first_line.strip()))
                                end = float(time_utils.get_sec_h_m_s(thrid_line.strip()))
                                sample_path = os.path.join(src_path.replace('/txt', '/audio_16kHz'), audio_file)
                                speaker_id = 'unknown'
                                sample_id =  file.replace('.txt', '') + '_{0}_{1}'.format(start, end)
                                channel = 1
                                transcription = text_utils.normalize_transcript(second_line.strip())
                                audio_length = end - start
                                tsv_list.append([sample_id, sample_path, channel, audio_length, start, end, transcription, speaker_id, args.database_name])

        df = pd.DataFrame(tsv_list, columns=['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database'])
        df.to_csv(tsv_file_name, sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate tsv from txt transcripts obtained from youtube")
    parser.add_argument('--src_path', help="path with source txt files", default="")
    parser.add_argument("--dst_path", help="path where to place tsv files", default="")
    parser.add_argument("--database_name", help="database name", default="")
    args = parser.parse_args()

    main(args)