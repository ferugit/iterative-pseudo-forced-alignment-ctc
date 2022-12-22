import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio

def main(args):

    if(not os.path.isfile(args.src)):
        print('Non valid source file')
    elif(not os.path.isdir(args.dst)):
        print('Non destination path')
    else:

        df = pd.read_csv(args.src, header=0, sep='\t')
        progress_bar = tqdm(total=len(df.index), desc='generating audio clips')

        for index, row in df.iterrows():
            
            # Check if already exist file:
            if os.path.isfile(os.path.join(args.dst, row['Sample_ID'] + '.wav')):
                print(os.path.join(args.dst, row['Sample_ID'] + '.wav') + ' already exist. Skipping...')
                progress_bar.update(1)
                continue

            info = torchaudio.info(row['Sample_Path'])
            audio, sr = torchaudio.load(
                row['Sample_Path'],
                frame_offset=int(row['Start']*info.sample_rate),
                num_frames=int((row['End']-row['Start'])*info.sample_rate)    
            )
            if audio.shape[0] > 1:
                audio = torch.mean(audio, 0)
                audio = audio.unsqueeze(0)
            torchaudio.save(os.path.join(args.dst, row['Sample_ID'] + '.wav'), audio, sr)
            progress_bar.update(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate clips from tsv")
    parser.add_argument('--src', help="source tsv file", default="")
    parser.add_argument("--dst", help="destination directory of cropped audios", default="")
    args = parser.parse_args()

    main(args)