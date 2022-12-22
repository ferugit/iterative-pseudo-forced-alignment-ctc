import os
import argparse
import pandas as pd
from tqdm import tqdm

import torchaudio

from speechbrain.pretrained import EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation

from utils.text_utils import normalize_transcript


def main(args):

    # Load ASR model
    source_path = 'config/'
    hparams_path = 'ctc_sp_with_wav2vec.yaml'
    savedir_path = 'data/asr/ctc/savedir'
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path) 

    # Segmentation tool
    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, time_stamps="fixed")

    if(args.commonvoice):

        df = pd.read_csv(args.commonvoice_tsv, header=0, sep='\t')
        segmented_list = []
        progress_bar = tqdm(total=len(df.index), desc='CommonVoice filtered words')

        for index, row in df.iterrows():
            audio_name = row['path']
            audio_path = os.path.join(args.commonvoice_data, audio_name)
            # try to load audio file
            info = torchaudio.info(audio_path)
            audio, sr = torchaudio.load(
                audio_path,
                channels_first=False
            )
            audio_length = info.num_frames/info.sample_rate
            normalized = asr_model.audio_normalizer(audio, sr)
            wanted_text = row['wanted_word']
            sentence = row['sentence']
            # Assume that different clients are different speakers
            speaker_id = row['client_id']
            list_sentence = normalize_transcript(sentence).upper().split(" ")
            sentence_to_align = []

            for i in range(1,  2*len(list_sentence)):
                sentence_to_align.append("·" if(i+1)%2 else list_sentence[int(i/2)])
        
            # Word-level alignment
            segments = aligner(normalized, sentence_to_align, name="CommonVoiceSP")
            segments = str(segments).split("\n") 

            for index in range(len(segments)):
                segment = segments[index].split(" ")

                if(segment[-1] == wanted_text):
                    segment_start = float(segment[2]) + args.offset_time + args.left_offset
                    segment_end = float(segment[3]) + args.offset_time + args.right_offset
                    segment_score = float(segment[4])
                    segment_length = segment_end - segment_start
                    sample_id = "_".join([audio_name.replace('.mp3', ''), str(segment_start), str(segment_end)])

                    if(args.store_clips and args.cropped and (index < args.limit)):
                        audio, sr = torchaudio.load(
                            audio_path,
                            frame_offset=int(segment_start*info.sample_rate),
                            num_frames=int((segment_end-segment_start)*info.sample_rate)    
                        )
                        torchaudio.save(os.path.join(args.cropped, audio_name), audio, sr)

                    segmented_list.append([sample_id, audio_path, audio_length, segment_length, segment_start, segment_end, segment_score, sentence, speaker_id, wanted_text.lower()])

            progress_bar.update(1)

        segmented_df = pd.DataFrame(segmented_list, columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Segment_Length', 'Start', 'End','Segment_Score', 'Transcription', 'Speaker_ID', 'Word'])
        segmented_df['Database'] = 'CommonVoice_7.0'
        segmented_df.to_csv(os.path.join(args.commonvoice_dst, 'segmented_ctc.tsv'), sep = '\t', index=None)

    if(args.rtve2018):

        # TODO: once we have the utterance-level segmentation, continue with word-level
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate word-level segmentation")

    # CommonVoice
    parser.add_argument('--use_commonvoice', dest='commonvoice', action='store_true', help='segment commonvoice filtered dataset')
    parser.add_argument("--commonvoice_tsv", help="metadata with filtered audio", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/filtered/filtered.tsv")
    parser.add_argument("--commonvoice_data", help="directory of audio dataset", default="/disk3/data/fernandol/data/commonvoice/cv-corpus-7.0-2021-07-21/es/clips_16kHz")
    parser.add_argument("--commonvoice_dst", help="destination directory of filtered audios", default="data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips")

    # RTVE2018
    parser.add_argument('--use_rtve2018', dest='rtve2018', action='store_true', help='segment rtve2018 filtered dataset')
    parser.add_argument("--rtve2018_tsv", help="metadata with filtered audio", default="data/RTVE2018/filtered/filtered.tsv")
    parser.add_argument("--rtve2018_dst", help="destination directory of filtered audios", default="data/RTVE2018/segmented_clips")
    
    # Deviation
    parser.add_argument('--offset_time', type=float, default=0.0, metavar='offset', help='time window covered by every data sample')
    parser.add_argument("--left_offset", type=float, default=0.0, metavar='left_offset', help='left offset to correct deviation')
    parser.add_argument("--right_offset", type=float, default=0.0, metavar='right_offset', help='right offset to correct deviation')
    
    # Store clips
    parser.add_argument('--store_clips', dest='store_clips', action='store_true', help='store segmented clips')
    parser.add_argument("--cropped", help="destination directory of cropped audios", default="data/audio/segmented/ctc")
    parser.add_argument('--limit', type=int, default=-1, metavar='N', help='number of files to segment')
    args = parser.parse_args()

    # Run main
    main(args)