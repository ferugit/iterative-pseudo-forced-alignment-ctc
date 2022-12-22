from operator import index
import os
from tqdm import tqdm
import argparse

import torch
import torchaudio

import pandas as pd

import speechbrain as sb
from speechbrain.pretrained import EncoderASR


def main(args):

    if(os.path.isfile(args.tsv)):

        # Load ASR model
        source_path = 'config/'
        hparams_path = 'ctc_sp_with_wav2vec.yaml'
        savedir_path = 'data/asr/ctc/savedir'
        asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)
        asr_model.to('cuda:2')
        device = asr_model.mods.encoder.wav2vec2.model.device
        ckpt = asr_model.hparams.exp_path

        df = pd.read_csv(args.tsv, header=0, sep='\t')

        # filter by shows
        #dev2_df = pd.concat([df[df['Sample_ID'].str.contains('AFI')].reset_index(), df[df['Sample_ID'].str.contains('LT24HEco')].reset_index()], axis=0)
        dev2_df = df

        progress_bar = tqdm(total=len(df.index), desc='RTVE hypothesis')

        results_list = []
        
        for idx, row in df.iterrows():
            original_file = row["Sample_Path"]
            reference_text = row["Transcription"]
            sample_id = row['Sample_ID']
            
            audio_path = '/disks/md1-8T/users/cx02275/data/Albayzin-aligned/clips/' + sample_id + '.wav'

            # Load audio
            audio, sr = torchaudio.load(audio_path, channels_first=False)
            normalized = asr_model.audio_normalizer(audio, sr)
            #encoded = asr_model.encode_batch(normalized.unsqueeze(0).to(device), torch.Tensor([1.0]).to(device))
            wavs, wav_lens = normalized.unsqueeze(0).to(device), torch.Tensor([1.0]).to(device)
            encoded = asr_model.mods.encoder(wavs, wav_lens)
            p_ctc = asr_model.hparams.log_softmax(encoded)
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, torch.Tensor([1.0]), blank_id=asr_model.hparams.blank_index
            )
            hypothesis = asr_model.tokenizer.decode_ids(sequence[0])

            results_list.append([sample_id, hypothesis, reference_text ])

            progress_bar.update(1)

        hypothesis_df = pd.DataFrame(results_list, columns=['Sample_ID', 'Hypothesis', 'Reference'])
        hypothesis_df.to_csv(ckpt + '_hypothesis.tsv', sep='\t', index=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script generate hypothesis vs reference")
    parser.add_argument("--tsv", help="tsv file with clips", default="")
    args = parser.parse_args()

    main(args)