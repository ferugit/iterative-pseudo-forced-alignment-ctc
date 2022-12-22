import os
import argparse

import pandas as pd


def main(args):
    
    if(os.path.isfile(args.tsv)):

        print('Filtering {0}'.format(args.tsv))
        print('Threshold {0}'.format(args.minimum_score))
        
        partition_name = args.tsv.split('/')[-1].replace('.tsv', '')
        df = pd.read_csv(args.tsv, header=0, sep='\t')
        filtered_df = df[df['Segment_Score'] > args.minimum_score]
        filtered_df['Start'] += args.offset_time
        filtered_df['End'] += args.offset_time
        filtered_df['Start'] += args.right_offset
        filtered_df['End'] += args.left_offset
        filtered_df['Audio_Length'] = filtered_df['End'] - filtered_df['Start']
        # REMOVE: this code will not be needed in the future
        filtered_df['Sample_ID'] = filtered_df['Sample_ID'].apply(lambda x: x.split('/')[-1].replace('.wav', ''))
        
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
    parser.add_argument('--minimum_score', type=float, default=-1.0, metavar='TH', help='minimum threshold')

    # Deviation
    parser.add_argument('--offset_time', type=float, default=0.0, metavar='offset', help='time window covered by every data sample')
    parser.add_argument("--left_offset", type=float, default=0.0, metavar='left_offset', help='left offset to correct deviation')
    parser.add_argument("--right_offset", type=float, default=0.0, metavar='right_offset', help='right offset to correct deviation')
    args = parser.parse_args()

    # Run main
    main(args)