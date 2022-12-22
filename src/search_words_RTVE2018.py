import os
import json
import argparse

import pandas as pd


def main(args):

    # Read configuration file: wanted words
    wanted_words = json.load(open(args.config_file, 'r'))["words"]

    # Common columns
    columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database']

    # Filetered dataframe
    filtered_df = pd.DataFrame()
    wanted_word_list = []

    # Read train metadata
    train_df_path = os.path.join(args.src, 'train.tsv')
    dev1_df_path = os.path.join(args.src, 'dev1.tsv')
    dev2_df_path = os.path.join(args.src, 'dev2.tsv')

    train_df = pd.read_csv(train_df_path, header=0, sep='\t', usecols=columns)
    dev1_df = pd.read_csv(dev1_df_path, header=0, sep='\t', usecols=columns)
    dev2_df = pd.read_csv(dev2_df_path, header=0, sep='\t', usecols=columns)

    df = pd.concat([train_df, dev1_df, dev2_df], ignore_index=True)
    
    # Drop rows with NaN values
    df = df.dropna()

    # Look for text: uppercase and lowercase
    for word in wanted_words:
        row_count = len(filtered_df.index)
        filtered_df = filtered_df.append(df[df['Transcription'].str.contains(word + ',')])
        filtered_df = filtered_df.append(df[df['Transcription'].str.contains(word + ' ')])
        filtered_df = filtered_df.append(df[df['Transcription'].str.contains(word.capitalize() + ',')])
        filtered_df = filtered_df.append(df[df['Transcription'].str.contains(word.capitalize() + ' ')])
        new_row_count = len(filtered_df.index)
        wanted_word_list += ([word.upper()] * (new_row_count - row_count))
        print('Found: {0} occurrences of word: {1}'.format(new_row_count - row_count, word.upper()))
    
    filtered_df['wanted_word'] = wanted_word_list
    filtered_df.to_csv(os.path.join(args.dst, 'filtered.tsv'), sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to search wanted words in RTVE2018")
    parser.add_argument("--src", help="source directory of metadata", default="data/RTVE2018/tsv/")
    parser.add_argument("--dst", help="destination directory of metadata", default="data/RTVE2018/filtered/")
    parser.add_argument("--config_file", help="configuration file", default="config/words.json")
    args = parser.parse_args()

    # Run main
    main(args)