import argparse
import pandas as pd
from tqdm import tqdm

import torchaudio

from speechbrain.pretrained import EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation

from utils.text_utils import normalize_transcript
from utils.alignment_utils import alignment_logger


def main(args):

    log_name = args.tsv_path.split('/')[-1].replace('.tsv', '')
    logger = alignment_logger(args.logs_path, f"{log_name}")
    logger.debug('Starting word alignment for file: ' + str(args.tsv_path))

    # Load ASR model
    source_path = args.asr_src_path
    hparams_path = args.asr_yaml
    savedir_path = args.asr_savedir
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path) 

    # Segmentation tool
    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, time_stamps="fixed")

    # Read source information
    df = pd.read_csv(args.tsv_path, header=0, sep='\t')

    # Start alignment
    segmented_list = []
    progress_bar = tqdm(total=len(df.index), desc='words-level alignment')

    for _, row in df.iterrows():
        audio_path = row['Sample_Path']
        audio_name = audio_path.split('/')[-1]
        audio_extension = audio_name.split('.')[-1]
        clip_start =  float(row['Start'])
        clip_end = float(row['End'])
        clip_length = clip_end - clip_start
        wanted_text = row['Wanted_Text']
        database = row['Database']
        sentence = row['Normalized_Transcription']
        speaker_id = row['Speaker_ID']

        # Read audio and normalize
        try:
            info = torchaudio.info(audio_path)
            audio, sr = torchaudio.load(
                audio_path,
                frame_offset=int(clip_start * info.sample_rate),
                num_frames=int(clip_length * info.sample_rate), 
                channels_first=False
            )
            audio_normalized = asr_model.audio_normalizer(audio, sr)
        except:
            print('Start frame: {0}. Enf frame: {1}. Row: {2}'.format(clip_start, clip_end, row))

        # Prepare text to align
        list_sentence = normalize_transcript(sentence).upper().split(wanted_text)
        list_sentence.insert(1, wanted_text)
        
        # Remove empty strings: if wanted text is at the end/beggining of the utterance
        while("" in list_sentence):
            list_sentence.remove("")
        
        # Trim empty spaces
        list_sentence = [sub_utterance.strip() for sub_utterance in list_sentence]

        # Prepare sentence to be aligned
        sentence_to_align = []

        for i in range(1,  2*len(list_sentence)):
            sentence_to_align.append("·" if(i+1)%2 else list_sentence[int(i/2)])
        sentence_to_align.insert(len(sentence_to_align), "·")
        
        # Get AM posteriors
        lpz = aligner.get_lpz(audio_normalized)
        
        # Conflate text & lpz & config as a segmentation task object
        task = aligner.prepare_segmentation_task(
            sentence_to_align,
            lpz,
            row['Sample_ID'],
            audio_normalized.shape[0]
            )
        
        # Apply CTC alignment
        segments = aligner.get_segments(task)
        task.set(**segments)
        segments = task.__str__().strip().split("\n")                
        list_of_segments = [segment.split(" ", 5) for segment in segments]  

        for segment in list_of_segments:
            
            if(len(segment) != 6):
                logger.debug('Some problem with segment: ' + str(segment))
                continue

            if(segment[-1] == wanted_text):
                segment_start = float(segment[2]) + args.offset_time + args.left_offset
                segment_end = float(segment[3]) + args.offset_time + args.right_offset
                segment_score = float(segment[4])
                segment_length = segment_end - segment_start
                absolute_start = clip_start + segment_start
                absolute_end = clip_start + segment_end
                sample_id = "_".join([audio_name.replace(audio_extension, ''), str(absolute_start), str(absolute_end)])

                logger.debug('{0} | {1} | {2} | {3}'.format(
                            round(absolute_start, 3),
                            round(absolute_end, 3),
                            round(segment_score, 3),
                            segment[-1]
                            )
                        )

                segmented_list.append([sample_id, audio_path, segment_length, absolute_start, absolute_end, segment_score, sentence, speaker_id, wanted_text.lower(), database])
        
        progress_bar.update(1)

    segmented_df = pd.DataFrame(
        segmented_list,
        columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Start', 'End','Segment_Score', 'Transcription', 'Speaker_ID', 'Word', 'Database']
        )
    segmented_df.to_csv(args.tsv_path.replace('_filtered.tsv', '_words.tsv'), sep = '\t', index=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate word-level segmentation")

    # ASR arguments
    parser.add_argument("--asr_src_path", help="ASR source path", default="")
    parser.add_argument("--asr_yaml", help="ASR yaml file", default="")
    parser.add_argument("--asr_savedir", help="ASR save dir to store a symbolic link", default="")

    # Sosurce data
    parser.add_argument("--tsv_path", help="metadata with filtered audio", default="")
    parser.add_argument("--dst_path", help="path to place results", default="")

    # Deviation
    parser.add_argument('--offset_time', type=float, default=0.0, metavar='offset', help='temporal shift in seconds of alignment (left and rigth)')
    parser.add_argument("--left_offset", type=float, default=0.0, metavar='left_offset', help='left offset in seconds')
    parser.add_argument("--right_offset", type=float, default=0.0, metavar='right_offset', help='right offset in seconds')
    parser.add_argument('--collar', type=float, default=0.0, metavar='collar', help='collar to apply to alignment in seconds')

    # Logs
    parser.add_argument("--logs_path", help="path to place logs", default="")

    args = parser.parse_args()

    # Run main
    main(args)