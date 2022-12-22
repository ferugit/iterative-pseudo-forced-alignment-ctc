import os
import argparse
import pandas as pd
from tqdm import tqdm

import torchaudio

from speechbrain.pretrained import EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation


def main(args):

    # Load ASR model
    source_path = 'config/'
    hparams_path = 'ctc_sp_with_wav2vec.yaml'
    savedir_path = 'data/asr/ctc/savedir'
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path)
    asr_model.to('cuda:0')
    asr_model.device = asr_model.mods.encoder.wav2vec2.model.device
    device = asr_model.mods.encoder.wav2vec2.model.device

    # Segmentation tool
    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, time_stamps="fixed")

    df = pd.read_csv(args.tsv, header=0, sep='\t')
    progress_bar = tqdm(total=len(df.index), desc='generate word level segmentation')

    fusion_path = "/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/tid_word_hypothesis"

    with open(fusion_path, "w") as results_file:

        for index, row in df.iterrows():
            audio_path = row['Sample_Path']
            clip_start = float(row['Start'])
            clip_end = float(row['End'])
            clip_length =  clip_end - clip_start
            
            # try to load audio file
            info = torchaudio.info(audio_path)
            audio, sr = torchaudio.load(
                audio_path,
                frame_offset=int(clip_start * info.sample_rate),
                num_frames=int(clip_length * info.sample_rate), 
                channels_first=False
            )

            normalized = asr_model.audio_normalizer(audio, sr)
            uttid = row['Uttid']
            sentence_to_align = str(row['Hypothesis']).upper()
            sentence_to_align = sentence_to_align.split(" ") if sentence_to_align != "NAN" else ["·"] 
            #print(sentence_to_align)
        
            # Word-level alignment
            result = aligner(normalized, sentence_to_align, name=uttid)
            segments = str(result).strip().split("\n")

            list_of_segments = [segment.split(" ", 5) for segment in segments]
            corrected_segments = []

            for segment in list_of_segments:
                            
                if(len(segment) != 6):
                    raise Exception('Some problem with segment: ' + str(result))
                
                if segment[-1] == "·":
                    segment[-1] = ""
                else:    
                    segment[-1] = segment[-1].lower()
                    
                corrected_segments.append(" ".join(segment).strip())
            
            segments = "\n".join(corrected_segments)
            results_file.write(segments + "\n")
            #print(segments)

            progress_bar.update(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate word-level segmentation")

    # CommonVoice
    parser.add_argument("--tsv", help="metadata with hypothesis", default="")
    parser.add_argument("--dst", help="destination directory of segmentation files", default="")
    args = parser.parse_args()

    # Run main
    main(args)