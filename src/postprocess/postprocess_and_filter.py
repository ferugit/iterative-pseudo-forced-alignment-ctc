import os
import argparse
import pandas as pd

import torchaudio


def apply_start_collar(x, delta):
    if (x - delta) > 0.0:
        return x - delta
    else:
        return 0.0

def apply_end_collar(df, delta):
    old_audio_path = None
    list_of_end_times = []

    for _, row in df.iterrows():
        audio_path = row['Sample_Path']

        if old_audio_path != audio_path:
            info = torchaudio.info(audio_path)
            audio_length_time = info.num_frames/info.sample_rate
        
        end = float(row['End'])

        if (end + delta) > audio_length_time:
            new_end = audio_length_time
        else:
            new_end = end + delta

        old_audio_path = audio_path
        list_of_end_times.append(new_end)
    
    return list_of_end_times

    


def main(args):
    
    if(os.path.isfile(args.tsv)):

        print('Filtering {0}'.format(args.tsv))
        print('Threshold {0}'.format(args.minimum_score))
        
        partition_name = args.tsv.split('/')[-1].replace('.tsv', '')
        df = pd.read_csv(args.tsv, header=0, sep='\t')
        if (args.comp == 'gt'):
            filtered_df = df[df['Segment_Score'] >= args.score]
        elif (args.comp == 'lt'):
            filtered_df = df[df['Segment_Score'] < args.score]
        filtered_df['Start'] += args.offset_time
        filtered_df['End'] += args.offset_time
        filtered_df['Start'] += args.left_offset
        filtered_df['End'] += args.right_offset

        # apply collar
        if args.collar > 0.0:
            delta = args.collar/2
            filtered_df['Start'] = filtered_df['Start'].apply(lambda x: apply_start_collar(x, delta))
            list_of_end_times = apply_end_collar(filtered_df, delta)
            # New column times
            filtered_df.drop('End', axis=1, inplace=True)
            filtered_df['End'] = list_of_end_times

        filtered_df['Audio_Length'] = filtered_df['End'] - filtered_df['Start']
        audio_seconds = filtered_df['Audio_Length'].sum()

        print('Total audio length {0} seconds'.format(audio_seconds))

        filtered_df.to_csv(os.path.join(os.path.dirname(args.tsv), partition_name + '_filtered.tsv'), index=None, sep='\t')


    else:
        print('tsv file does not exist ({0})'.format(args.tsv))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script post-process RTVE2018DB aligned files")
    
    # Alignment TSV
    parser.add_argument("--tsv", help="tsv with alignment information", default="")

    # Post-process parameters
    parser.add_argument('--score', type=float, default=-1.0, metavar='TH', help='threshold')
    parser.add_argument('--comp', type=str, default="gt", metavar='comparation', choices=['gt', 'lt'], help='greater than (gt) or lesser than (lt)')

    # Deviation
    parser.add_argument('--offset_time', type=float, default=0.0, metavar='offset', help='temporal shift in seconds of alignment (left and rigth)')
    parser.add_argument("--left_offset", type=float, default=0.0, metavar='left_offset', help='left offset in seconds')
    parser.add_argument("--right_offset", type=float, default=0.0, metavar='right_offset', help='right offset in seconds')
    parser.add_argument('--collar', type=float, default=0.0, metavar='collar', help='collar to apply to alignment in seconds')

    args = parser.parse_args()

    # Run main
    main(args)