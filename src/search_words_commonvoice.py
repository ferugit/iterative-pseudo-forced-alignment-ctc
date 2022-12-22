import os
import json
import argparse

import pandas as pd


def main(args):

    # Read configuration file: wanted words
    wanted_words = json.load(open(args.config_file, 'r'))["words"]

    # Filetered dataframe
    filtered_df = pd.DataFrame()
    wanted_word_list = []

    # Read dataset metadata
    df_path = os.path.join(args.src, 'validated.tsv')
    df = pd.read_csv(df_path, header=0, sep='\t')

    # Look for text: uppercase and lowercase
    for word in wanted_words:
        row_count = len(filtered_df.index)
        filtered_df = filtered_df.append(df[df['sentence'].str.contains(word + ',')])
        filtered_df = filtered_df.append(df[df['sentence'].str.contains(word + ' ')])
        filtered_df = filtered_df.append(df[df['sentence'].str.contains(word.capitalize() + ',')])
        filtered_df = filtered_df.append(df[df['sentence'].str.contains(word.capitalize() + ' ')])
        new_row_count = len(filtered_df.index)
        wanted_word_list += ([word.upper()] * (new_row_count - row_count))
    
    filtered_df['wanted_word'] = wanted_word_list
    counts = filtered_df['wanted_word'].value_counts()
    print('Found following occurrences: \n' + str(counts))
    filtered_df.drop_duplicates(keep='first', inplace=True)
    filtered_df.to_csv(os.path.join(args.dst, 'filtered.tsv'), sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to search wanted words in CommonVoice")

    # Source Dns data placed in the data folder of the project 
    parser.add_argument("--src", help="source directory of metadata", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/metadata/")
    parser.add_argument("--dst", help="destination directory of metadata", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/filtered/")
    parser.add_argument("--config_file", help="configuration file", default="config/words.json")
    args = parser.parse_args()

    # Run main
    main(args)