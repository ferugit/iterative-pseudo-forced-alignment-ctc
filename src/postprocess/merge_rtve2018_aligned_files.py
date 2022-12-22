import os
import argparse

import pandas as pd


def main(args):

    if(os.path.isfile(args.partition) and os.path.isdir(args.src)):
        
        partition_name = args.partition.split('/')[-1].replace('.tsv', '')
        partition_df = pd.read_csv(args.partition, header=0, sep='\t')
        audio_paths = partition_df['Sample_Path'].unique()

        list_of_dataframes = []

        for audio_path in audio_paths:
            aligned_filename = os.path.join(args.src, audio_path.split('/')[-1].replace('.wav', '.tsv'))

            if(os.path.isfile(aligned_filename)):
                aligned_df = pd.read_csv(aligned_filename, header=0, sep='\t')
                list_of_dataframes.append(aligned_df)

        aligned_partition_df = pd.concat(list_of_dataframes, ignore_index=True)
        aligned_partition_df.to_csv(os.path.join(args.src, partition_name + '_aligned.tsv'), index=None, sep='\t')
        
    else:
        print('Partition file or source directory does not exist')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script merge RTVE aligned files")
    parser.add_argument("--partition", help="partition that has been aligned", default="")
    parser.add_argument("--src", help="folder with aligned files", default="")
    args = parser.parse_args()

    # Run main
    main(args)