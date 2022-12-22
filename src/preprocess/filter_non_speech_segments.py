import os
import argparse

import pandas as pd


def main(args):

    if(os.path.isfile(args.src) and os.path.isdir(args.dst)):
        df = pd.read_csv(args.src, header=0, sep='\t')
        audio_paths = df['Sample_Path'].unique()

        # Get previous non-speech segments length
        previous_start = 0.0
        for index, row in df.iterrows():
            non_speech_length = row['Start'] - previous_start
            df.at[index,'Non_Speech_Length'] = non_speech_length
            previous_start = row['End']

        results_list = []
        for path in audio_paths:
            file_df = df[df['Sample_Path'] == path]
            audio_length = float(file_df.sample(1)['Audio_Length'])
            filtered_df = file_df[file_df['Non_Speech_Length'] >= args.length]
            
            if filtered_df.empty:
                results_list.append([path, audio_length, 0.0, audio_length, audio_length])
            else:
                filtered_df = filtered_df.reset_index()
                previous_start = 0.0
                df_len = len(filtered_df.index)

                for index, row in filtered_df.iterrows():
                    end = row['Start'] - row['Non_Speech_Length']
                    segment_length = end - previous_start
                    results_list.append([path, audio_length, previous_start, end, segment_length])
                    previous_start = row['Start']

                    if(index == df_len - 1):
                        segment_length = audio_length - previous_start
                        results_list.append([path, audio_length, previous_start, audio_length, segment_length])
        
        columns = df.columns.tolist()
        columns.remove('Non_Speech_Length')
        results_df = pd.DataFrame(results_list, columns=columns)
        results_df.to_csv(os.path.join(args.dst, args.src.split('/')[-1].replace('.tsv', '_filtered.tsv')), sep='\t', index=None)

    else:
        print('Source file or destination directory does not exist.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to filter VAD segments with length criteria")
    
    parser.add_argument("--src", help="metadata with information to filter", default="")
    parser.add_argument("--dst", help="path to place the results", default="")
    parser.add_argument('--length', type=float, default=30.0, help='minimum size to consider non-speech segment (seconds)')

    args = parser.parse_args()

    # Run main
    main(args)