import torchaudio

import pandas as pd


if __name__ == "__main__":

    test_path = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2020DB/tsv/test.tsv'

    df = pd.read_csv(test_path, header=0, sep='\t')
    audio_paths = df['Sample_Path'].unique().tolist()

    for path in audio_paths:
        audio_metadata = torchaudio.info(path)
        full_audio_length = audio_metadata.num_frames/audio_metadata.sample_rate
        df.loc[df['Sample_Path'] == path, 'Audio_Length'] = full_audio_length
    
    df['Show'] = df['Sample_ID'].str.split('-').str[0]
    df = df.drop_duplicates(subset='Sample_Path', keep="last")

    grouped_df = df.groupby(['Show']).sum()
    grouped_df.drop(['Channel', 'Start', 'End'], axis=1, inplace=True)
    grouped_df['Audio_Length_h'] = grouped_df['Audio_Length'] / 3600
    print(grouped_df)
    print("Total hours: ", grouped_df['Audio_Length_h'].sum())
        
