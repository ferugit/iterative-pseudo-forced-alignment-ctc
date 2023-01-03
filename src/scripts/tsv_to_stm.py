import os
import argparse

import pandas as pd


def main(args):

    if ((not os.path.isdir(args.src_path)) or ((not os.path.isdir(args.dst_path)))):
        print("Non valid alrguments to convert tsv files to stm.")

    else:
        for root, _, files in os.walk(os.path.join(args.src_path)):

            for file in files:
                    
                    if file.endswith('.tsv') and not file.startswith('.'):
                        stm_file_name = os.path.join(args.dst_path, file).replace('.tsv', '.stm')
                        src_df = pd.read_csv(os.path.join(args.src_path, file), header=0, sep='\t')
                        filename_wo_extension = file.replace('.tsv', '')

                        with open(stm_file_name, 'w') as stm_file:
                            
                            for index, row in src_df.iterrows():
                                

                                channel = row['Channel']
                                speaker_id = row['Speaker_ID']
                                start = round(row['Start'], 3)
                                end = round(row['End'], 3)
                                transcription = row['Transcription'].lower()

                                stm_line = "{waveform} {channel} {speaker_id} {start} {end} <,,> {transcription}\n".format(
                                    waveform=filename_wo_extension,
                                    channel=channel,
                                    speaker_id=speaker_id,
                                    start=start,
                                    end=end,
                                    transcription=transcription
                                )

                                stm_file.write(stm_line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate clips from tsv")
    parser.add_argument('--src_path', help="path with source tsv files", default="")
    parser.add_argument("--dst_path", help="path where to place stm files", default="")
    args = parser.parse_args()

    main(args)