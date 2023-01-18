import os
import json
import argparse
import pandas as pd

import utils.text_utils as text_utils


def main(args):

    # Read configuration file: wanted words
    wanted_words = json.load(open(args.config_file, 'r'))["words"]

    # Filetered dataframe
    filtered_df = pd.DataFrame()
    wanted_word_list = []

    # Read dataset metadata
    df_path = os.path.join(args.tsv_path)
    df = pd.read_csv(df_path, header=0, sep='\t')

    if wanted_words != ["*"]:
        # Normalize text
        df['Normalized_Transcription'] = df[args.text_column].apply(lambda x: text_utils.normalize_transcript(x).upper())

        # Look for text: uppercase and lowercase
        for word in wanted_words:
            row_count = len(filtered_df.index)
            filtered_df = filtered_df.append(df[df['Normalized_Transcription'].str.contains(word.upper())])
            new_row_count = len(filtered_df.index)
            wanted_word_list += ([word.upper()] * (new_row_count - row_count))
        
        filtered_df['Wanted_Text'] = wanted_word_list
    else:
        # Wanted words are all words contained in Transcription column
        df['Normalized_Transcription'] = df[args.text_column].apply(lambda x: text_utils.normalize_transcript(x).upper())
        df = df[df["Normalized_Transcription"].str.contains("UNKNOWN") == False]
        df = df[df["Normalized_Transcription"].str.contains("UNTRANSCRIBED") == False]
        df['Wanted_Text'] = df['Normalized_Transcription']
        filtered_df = df

    counts = filtered_df['Wanted_Text'].value_counts()
    print('Found following occurrences: \n' + str(counts))
    filtered_df.drop_duplicates(keep='first', inplace=True)
    tsv_name = args.tsv_path.split('/')[-1].replace('.tsv', '')
    filtered_df.to_csv(os.path.join(args.dst, tsv_name + '_filtered.tsv'), sep='\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to search wanted words in a tsv file")
    parser.add_argument("--tsv_path", help="source directory of metadata", default="")
    parser.add_argument("--dst", help="destination directory of metadata", default="")
    parser.add_argument("--config_file", help="configuration file", default="config/words.json")
    parser.add_argument("--text_column", help="column name that contains the text reference", default="Transcription")

    args = parser.parse_args()

    # Run main
    main(args)