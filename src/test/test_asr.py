import os
import argparse

import torch
import torchaudio

import speechbrain as sb
from speechbrain.pretrained import EncoderASR


def main(args):

    if(os.path.isfile(args.file)):
        
        # Load audio
        audio, sr = torchaudio.load(args.file, channels_first=False)
        print('Original audio chape: ' + str(audio.shape))
        audio = audio[:int(args.max_len*sr)]
        print('Cropped audio chape: ' + str(audio.shape))
        print('Audio file: ' + str(args.file) + '. Sampling rate: ' + str(sr))

        # Load ASR model
        source_path = 'config/'
        hparams_path = 'ctc_sp_with_wav2vec.yaml'
        savedir_path = 'data/asr/ctc/savedir'
        asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)

        # Normalize audio format
        normalized = asr_model.audio_normalizer(audio, sr)
        print('Normalized audio shape: ' + str(normalized.shape))
        print('Maximum absolute value of the signal: ' + str(torch.abs(normalized).max()))

        #text_array, _ = asr_model.transcribe_batch(normalized.unsqueeze(0), torch.Tensor([1.0]))
        print(asr_model.hparams.exp_path)
        encoded = asr_model.encode_batch(normalized.unsqueeze(0), torch.Tensor([1.0]))
        p_ctc = asr_model.hparams.log_softmax(encoded)
        # Decode token terms to words
        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, torch.Tensor([1.0]), blank_id=asr_model.hparams.blank_index
        )
        print(sequence)
        text_array = asr_model.tokenizer.decode_ids(sequence[0])
        print('Transcription is: ' + str(text_array))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script test ASR with a single audio")
    parser.add_argument("--file", help="audio file to get transcription", default="data/audio/test/common_voice_es_19694706.wav")
    parser.add_argument('--max_len', type=float, default=20.0, metavar='ML', help='maximum audio length: cropped if longer')
    args = parser.parse_args()

    main(args)