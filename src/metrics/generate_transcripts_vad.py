import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
from tqdm import tqdm
import argparse

import torch
import torchaudio

import pandas as pd

import speechbrain as sb
from speechbrain.pretrained import EncoderASR


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


def get_text(asr_model, info, audio_path, clip_start, clip_end):

    # Load audio
    clip_length = clip_end - clip_start
    audio, sr = torchaudio.load(
        audio_path,
        frame_offset=int(clip_start* info.sample_rate),
        num_frames=int(clip_length * info.sample_rate), 
        channels_first=False
    )
    device = asr_model.mods.encoder.wav2vec2.model.device

    # Normalize audio format
    normalized = asr_model.audio_normalizer(audio, sr)
    normalized = normalized.float()
    wavs, wav_lens = normalized.unsqueeze(0).to(device), torch.Tensor([1.0]).to(device)
    encoded = asr_model.mods.encoder(wavs, wav_lens)
    #encoded = asr_model.encode_batch(normalized.unsqueeze(0).to(device), torch.Tensor([1.0]).to(device))
    p_ctc = asr_model.hparams.log_softmax(encoded)
    
    # Decode token terms to words
    sequence = sb.decoders.ctc_greedy_decode(
        p_ctc, torch.Tensor([1.0]), blank_id=asr_model.hparams.blank_index
    )
    text_array = asr_model.tokenizer.decode_ids(sequence[0])
    return text_array

def main(args):

    # Scan RTVE2018DB test
    check_path(args.dst)

    if os.path.isfile(args.tsv):

        # Load ASR model
        source_path = 'config/'
        hparams_path = 'ctc_sp_with_wav2vec.yaml'
        savedir_path = 'data/asr/ctc/savedir'
        asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)
        asr_model.to('cuda:2')
        print('Using ckpt: ', asr_model.hparams.exp_path)
        #asr_model.to('cpu')

        df = pd.read_csv(args.tsv, header=0, sep='\t')
        audio_paths = df['Sample_Path'].unique()
        progress_bar = tqdm(total=len(audio_paths), desc='RTVE test')

        # Loop each audio in chunks for the sake of computational load
        for audio_path in audio_paths:
            file_df = df[df['Sample_Path'] == audio_path]
            file_df.reset_index(drop=True, inplace=True)
            info = torchaudio.info(audio_path)
            print('Audio file: ' + str(audio_path))
            file_transcripts_list = []

            for index, row in file_df.iterrows():
                start = float(row['Start'])
                end = float(row['End'])
                segment_length = float(row['Segment_Length'])
                
                hypothesis_text = get_text(asr_model, info, audio_path, start, end).lower()
                file_transcripts_list.append(hypothesis_text.strip())
                    
            
            file_transcripts = ' '.join(file_transcripts_list)
            file_name = os.path.join(args.dst, audio_path.split('/')[-1].replace('.wav', '.txt'))
            with open(file_name, 'w') as f:
                f.write(file_transcripts.strip())

            progress_bar.update(1)
    
    else:
        print('{0} file does not exists, please create it.'.format(args.tsv))  


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to test asr")
    
    # RTVE2018
    parser.add_argument("--tsv", help="tsv with vad segemnts path", default="")
    parser.add_argument("--dst", help="resultant transcripts", default="data/RTVE2018DB/transcripts/test/")
    args = parser.parse_args()

    # Run main
    main(args)
