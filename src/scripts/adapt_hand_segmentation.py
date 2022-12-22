import os
import argparse

import torchaudio

import pandas as pd


def main(args):
    
    # Read hand segmentation
    handmade_df = pd.read_csv(args.src, header=0)
    handmade_df['path'] = handmade_df['name']
    handmade_df['path'] = handmade_df['name'].apply(
        lambda x: os.path.join(args.dataset, x)
        )

    if(args.cropped):        

        for _, row in handmade_df.iterrows():
            audio_name = row['name']
            audio_path = row['path']
            segment_start = row['start']
            segment_end = row['end']

            audio, sr = torchaudio.load(
                audio_path,
                frame_offset=int(segment_start*48000),
                num_frames=int((segment_end-segment_start)*48000)
            )
            torchaudio.save(os.path.join(args.cropped, audio_name), audio, sr)


    handmade_df.to_csv(os.path.join(args.dst, 'segmented_hand.tsv'), index=None, sep='\t')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate handmade segmentation of CommonVoice dataset")
    parser.add_argument("--src", help="metadata with filtered audio", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips/hand_segmentation.csv")
    parser.add_argument("--dataset", help="directory of audio dataset", default="/disk3/data/fernandol/data/commonvoice/cv-corpus-7.0-2021-07-21/es/clips")
    parser.add_argument("--dst", help="destination directory of filtered audios", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips")
    parser.add_argument("--cropped", help="destination directory of cropped audios", default="data/audio/segmented/hand")
    args = parser.parse_args()

    main(args)