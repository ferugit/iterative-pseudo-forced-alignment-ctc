import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio

import speechbrain as sb
from speechbrain.pretrained import EncoderASR


def get_cer_and_wer(hypothesis, reference):
    # Metrics
    wer_metric = sb.utils.metric_stats.ErrorRateStats()
    cer_metric = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
    
    # Add text
    wer_metric.append([0], [hypothesis], [reference])
    cer_metric.append([0], [hypothesis], [reference])
    
    # Calculate
    cer = round(cer_metric.summarize("error_rate"), 2)
    wer = round(wer_metric.summarize("error_rate"), 2)
    return cer, wer


def main(args):

    if(os.path.isfile(args.tsv) and os.path.exists(args.dst_path) and args.name):
        # Load ASR model
        source_path = 'config/'
        hparams_path = 'ctc_sp_with_wav2vec.yaml'
        savedir_path = 'data/asr/ctc/savedir'
        asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)
        asr_model.to('cuda:2')
        device = asr_model.mods.encoder.wav2vec2.model.device
        ckpt = asr_model.hparams.exp_path

        df = pd.read_csv(args.tsv, header=0, sep='\t')

        progress_bar = tqdm(total=len(df.index), desc='Aligment hypothesis')

        hypothesis_list = []
        cer_list = []
        wer_list = []
        
        for idx, row in df.iterrows():
            audio_path = row["Sample_Path"].replace('/disk1/audio/Albayzin_2022/', '/home/cx02775/data/Albayzin_2022/')

            # Load audio
            info = torchaudio.info(audio_path)
            audio, sr = torchaudio.load(
                audio_path,
                frame_offset=int(row['Start']*info.sample_rate),
                num_frames=int((row['End']-row['Start'])*info.sample_rate),
                channels_first=False  
            )

            # Pre-process audio
            normalized = asr_model.audio_normalizer(audio, sr)
            wavs, wav_lens = normalized.unsqueeze(0).to(device), torch.Tensor([1.0]).to(device)
            
            # Process audio
            encoded = asr_model.mods.encoder(wavs, wav_lens)
            p_ctc = asr_model.hparams.log_softmax(encoded)
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, torch.Tensor([1.0]), blank_id=asr_model.hparams.blank_index
            )
            hypothesis = asr_model.tokenizer.decode_ids(sequence[0])
            cer, wer = get_cer_and_wer(hypothesis, row['Transcription'])
            
            hypothesis_list.append(hypothesis)
            cer_list.append(cer)
            wer_list.append(wer)
            progress_bar.update(1)

        df['Hypothesis'] = hypothesis_list
        df['WER'] = wer_list
        df['CER'] = cer_list

        df.to_csv(os.path.join(args.dst_path, args.name), sep='\t', index=None)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script generate hypothesis vs reference")
    parser.add_argument("--tsv", help="tsv file with clips", default="")
    parser.add_argument("--dst_path", help="destination folder path", default="")
    parser.add_argument("--name", help="resultant tsv name", default="")
    args = parser.parse_args()

    main(args)