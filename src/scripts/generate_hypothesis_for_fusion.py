import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
from tqdm import tqdm

import torch
import torchaudio

import pandas as pd

import speechbrain as sb
from speechbrain.pretrained import EncoderASR

import torchaudio


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


def main():

    # Load ASR model
    source_path = 'config/'
    hparams_path = 'ctc_sp_with_wav2vec.yaml'
    #hparams_path = 'ctc_sp_with_wav2vec_87_tokenizer.yaml'
    savedir_path = 'data/asr/ctc/savedir'
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)
    print('Using ckpt: ', asr_model.hparams.exp_path)
    asr_model.to('cuda:1')
    #asr_model.to('cpu')

    # Read only RTVE2020DB segments
    segments_file_rtve2020 = '/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/rtve2020_test.v0.vadv0'
    segments_file_all = '/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/rtve2020_test.v0_eval2022.vadv0'

    audio_path_rtve2020 = '/disks/md1-8T/users/cx02275/data/Albayzin_2022/RTVE2020DB/test/audio_16kHz/S2T'
    audio_path_rtve2022 = '/disks/md1-8T/users/cx02275/data/Albayzin_2022/RTVE2022DB/test/audio_16kHz/S2T'

    fusion_path = "/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/tid_hypothesis"

    segments_list = []
    previous_show = ""

    with open(segments_file_all) as fp, open(fusion_path, "w") as results_file:
        lines = fp.readlines()
        
        for line in lines:
            line = line.split()

            uttid  = line[0]
            recording = line[1]
            start = float(line[2])
            end = float(line[3])
            
            show = recording.split("_")[1].lstrip("0")

            if "rtve2020" in uttid:
                audio_path = audio_path_rtve2020
            else:
                audio_path = audio_path_rtve2022

            sample_path = os.path.join(audio_path, show + ".wav")

            if "IM-271F8DG-C0026S01" in sample_path:
                sample_path = sample_path.replace("IM-271F8DG-C0026S01-", "IM-271F8DG_C0026S01-")

            if show != previous_show:
                audio_metadata = torchaudio.info(sample_path)

            hypothesis = get_text(asr_model, audio_metadata, sample_path, start, end).lower()

            results_file.write(uttid + " " + hypothesis + "\n")

            segments_list.append([sample_path, uttid, recording, show, start, end, end-start, hypothesis])

    columns = ["Sample_Path", "Uttid", "Recording", "Show", "Start","End",	"Segment_Length", "Hypothesis"]
    hypothesis_df = pd.DataFrame(segments_list, columns=columns)
    hypothesis_df.to_csv(fusion_path + ".tsv", index=None, sep="\t")


if __name__ == "__main__":
    main()