import os
import sys
import logging
import argparse
from tqdm import tqdm

import torchaudio
import pandas as pd

from utils import text_utils

from speechbrain.pretrained import EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation


def my_custom_logger(logger_name, split, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    format = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    logger.setLevel(level)
    logger.propagate = False
    
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(format)
    
    # Creating and adding the file handler
    file_handler = logging.FileHandler('logs/' + split + '/' + logger_name + '.log', mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def prepare_text(transcript, max_words_sequence=24):
    """
    Prepare text for alignment.
    If text is longer than max_word_sequence, then is splited.
    
    Arguments
    ---------
    transcript: new text to align
    max_words_sequence: maximum word length
    
    Returns
    -------
    transcript: list of text to align
    """
    if(len(transcript.split(' ')) > max_words_sequence):
        transcript = text_utils.split_long_transcript(
            transcript,
            max_words_sequence=max_words_sequence
            )
    else:
        transcript = [transcript] # transform to list
    # NOTE: Using the space between utterances provides worse results
    # list_to_align = []
    #for i in range(1,  2*len(transcript)):
        #list_to_align.append("·" if(i+1)%2 else transcript[int(i/2)])
        #transcript = list_to_align
    return transcript


def get_n_aligned_rows(list_of_splits, n_aligned_splits):
    index = 0
    for i in range(1, len(list_of_splits)):
        if(sum(list_of_splits[:-i]) <= n_aligned_splits):
            index = i
            break
    return len(list_of_splits[:-index])


def count_text_length(transcript):
    return len(" ".join(transcript))


def get_text_to_audio_proportion(audio_length, text_length, sample_rate):
    """
    Compare audio and text proportion.
    Time resolution is ~20ms. Mean phoneme duraction is 80ms
    margin_time: number of times to use for an utterance
    
    We assume that the audio duration will have as much 
    the duration of text_length * mean_phoneme_length * margin_time 
    Otherwise, we must discard some text fo rthe alignement or 
    read more audio

    Returns
    -------
    audio_to_text_proportion: proportion of duration of text inside the
        audio. 
        Values smaller than 1 mean that there is more audio than text.
        Values bigger than 1 mean that there is more text than audio.
    """
    margin_time = 3
    mean_phoneme_duration = 0.08 # 80ms
    maximum_text_duration_s = text_length * mean_phoneme_duration * margin_time
    maximum_text_duration_samples = maximum_text_duration_s * sample_rate
    return maximum_text_duration_samples / audio_length


def insert_row(idx, df, list_to_insert):
    # FIXME: attemt to keep a fixed order of the columns
    columns = ['Sample_ID', 'Sample_Path', 'Audio_Length', 'Start', 'End', 'Transcription', 'Speaker_ID', 'Database','Channel','Text_Length', 'Type']
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]
    df = dfA.append(pd.Series(list_to_insert, index=columns), ignore_index=True).append(dfB)
    return df.reset_index(drop=True)


def fix_time_reference(file_df, vad_file_df, real_audio_length, n_segments):
    """Fix time reference. Ignoring original reference information.
    This function provides an audio duration for each sentence 
    depending on the number of characters of the sentence.
    """
    file_df['Text_Length'] = file_df['Transcription'].astype(str).apply(lambda x: len(x))
    total_text_length = file_df['Text_Length'].sum()
    speech_length = vad_file_df['Segment_Length'].sum()
    audio_lenth_acc = 0.0
    vad_segment_counter = 0
    list_of_indexes = []

    for index, row in file_df.iterrows():
        row_text_length = row['Text_Length']
        text_percentage = row_text_length / total_text_length
        audio_legth = text_percentage * speech_length
        file_df.loc[index,'Start'] = audio_lenth_acc
        file_df.loc[index,'End'] = audio_lenth_acc + audio_legth
        audio_lenth_acc += audio_legth
        is_last_segment = True if((index + 1) == n_segments) else False
        
        # VAD speech segments
        speech_end = float(vad_file_df.iloc[[vad_segment_counter]]['End'])

        if audio_lenth_acc >= speech_end:
            file_df.loc[index,'End'] = speech_end
            
            if vad_segment_counter + 1 < len(vad_file_df.index):
                next_speech_start = float(vad_file_df.iloc[[vad_segment_counter + 1]]['Start'])
                audio_lenth_acc = next_speech_start
                vad_segment_counter += 1
                list_of_indexes.append(index)

        if is_last_segment and audio_lenth_acc < real_audio_length:
            file_df.loc[index,'End'] = real_audio_length
    
    file_df['Type'] = 'Speech'
    sample = file_df.sample(1)
    sample_path = str(sample['Sample_Path'].values[-1])
    channel = int(sample['Channel'].values[-1])
    database = str(sample['Database'].values[-1])

    for index in range(len(list_of_indexes)):
        current_speech_end = float(vad_file_df.iloc[[index]]['End'])
        next_speech_start = float(vad_file_df.iloc[[index + 1]]['Start'])

        # FIXME: stablish a fix order!
        row_new = [
            'Non-speech-' + str(index), sample_path, next_speech_start - current_speech_end, current_speech_end, 
            next_speech_start, "Non-Speech", "Non-Speech", database, channel, 0, 'Non-Speech'
            ]
        file_df = insert_row(list_of_indexes[index] + 1, file_df, row_new)

    # FIXME: remove following line
    #file_df.to_csv('references.tsv', sep='\t')
    return file_df


def find_a_valid_text_to_audio_proportion(audio_length, transcript, samples_to_frames_ratio):
    """
    Given a high quantity of text compared with the audio length.
    Find a valid text to audio proportion to calculate do the
    alignment.
    """
    # TODO: review this function: is not working that good!
    original_transcript = transcript
    max_number_of_chars = int(audio_length / samples_to_frames_ratio)
    new_discarded_transcripts = []
    
    for _ in range(1, len(transcript) + 1):
        
        current_text_length = count_text_length(transcript)
        print("Current text length: ", current_text_length)
        if current_text_length < max_number_of_chars:
            print('This happens')
            return transcript, new_discarded_transcripts
        
        new_discarded_transcripts.append(transcript[-1])
        transcript = transcript[:-1]
        print("Max number of chars: ", max_number_of_chars)

    return original_transcript, []
        

def fix_text_to_time_proportion(
    file_df, vad_file_df, real_audio_length, n_aligned, n_segments, last_anchor_time, logger
    ):
    """
    Reorganize text to time proportion, beacuse window size is big. 
    """
    # Length of remaining text & speech
    total_text_length = file_df.iloc[n_aligned:]['Text_Length'].sum()
    vad_file_df = vad_file_df[vad_file_df['End'] > last_anchor_time].reset_index(drop=True)
    vad_file_df.at[0, 'Start'] = last_anchor_time
    vad_file_df['Segment_Length'] = vad_file_df['End'] - vad_file_df['Start']
    speech_length = vad_file_df['Segment_Length'].sum()
    audio_lenth_acc = last_anchor_time
    vad_segment_counter = 0
    list_of_indexes = []
    logger.debug('Remaining audio: {0} | Remaining text: {1} | Remaining speech length {2}'.format(
        real_audio_length,
        total_text_length,
        speech_length
    ))
    logger.debug('Aligned index: {0}, Total segments: {1}'.format(n_aligned, n_segments))

    # Remove non-speech segments: by the moment
    n_non_speech_df = file_df[file_df['Type'] == 'Non-Speech']
    file_df = file_df[file_df['Type'] == 'Speech'].reset_index(drop=True)

    for index in range(n_aligned, n_segments):
        row = file_df.iloc[index]
        row_text_length = row['Text_Length']
        text_percentage = row_text_length / total_text_length
        audio_legth = text_percentage * speech_length
        file_df.loc[index,'Start'] = audio_lenth_acc
        file_df.loc[index,'End'] = audio_lenth_acc + audio_legth
        audio_lenth_acc += audio_legth
        is_last_segment = True if((index + 1) == n_segments) else False
        
        # VAD speech segments
        speech_end = float(vad_file_df.iloc[[vad_segment_counter]]['End'])

        if audio_lenth_acc >= speech_end:
            file_df.loc[index,'End'] = speech_end
            
            if vad_segment_counter + 1 < len(vad_file_df.index):
                next_speech_start = float(vad_file_df.iloc[[vad_segment_counter + 1]]['Start'])
                audio_lenth_acc = next_speech_start
                vad_segment_counter += 1
                list_of_indexes.append(index)

        if is_last_segment and audio_lenth_acc < real_audio_length:
            file_df.loc[index,'End'] = real_audio_length
    
    file_df['Type'] = 'Speech'
    sample = file_df.sample(1)
    sample_path = str(sample['Sample_Path'].values[-1])
    channel = int(sample['Channel'].values[-1])
    database = str(sample['Database'].values[-1])

    for index in range(len(list_of_indexes)):
        current_speech_end = float(vad_file_df.iloc[[index]]['End'])
        next_speech_start = float(vad_file_df.iloc[[index + 1]]['Start'])
        row_new = [
            'Non-speech-' + str(index), sample_path, channel, next_speech_start - current_speech_end, 
            current_speech_end, next_speech_start, "Non-Speech", "Non-Speech",
            database, 0, 'Non-Speech'
            ]
        file_df = insert_row(list_of_indexes[index] + 1, file_df, row_new)

    if(len(n_non_speech_df.index) > len(list_of_indexes)):
        generator = n_non_speech_df.iterrows()

        for i in range(len(n_non_speech_df.index) - len(list_of_indexes)):
            index, row = next(generator)
            file_df = insert_row(index, file_df, row)

    #file_df.to_csv('references_2.tsv', sep='\t')
    return file_df


def get_file_iterative_segmentation(
    asr_model, aligner, audio_path, file_df, vad_file_df,
    samples_to_frames_ratio, split):

    # Logging
    log_name = audio_path.split('/')[-1].replace('.wav', '')
    logger = my_custom_logger(f"{log_name}", split)
    logger.debug('Starting iterative alignment for file: ' + str(audio_path))

    # Alignment parameters
    threshold = -2.0 # good performance achieved with -6.0
    n_missaligned_utterances = 1 # consider that only one is due to bad transcription
    short_utterance_len = 30 # Minimum sequence of chars to select anchors
    max_words_sequence = 24 # Mesured from CommonVoice Dataset
    max_window_size = 70.0 # Seconds, for the sake of computational load
    window_to_stop = 500.0 # Seconds, windows to stop execution
    min_text_to_audio_prop = 0.8 # Min text to audio proportion
    max_text_to_audio_prop_exec = 10 # Number of consecutive exceptions to stop
    
    # Auxiliar tools to align
    new_segment_start = None # Anchor position
    discarded_transcripts = [] # List of discarded transcripts
    n_segments = len(file_df.index) # Total number of utterances
    list_of_splits = [] # Sub utterance aligned counter

    # Alignment results: already aligned utterances
    file_alignments = []

    # Audio information
    info = torchaudio.info(audio_path)
    labels_audio_length = float(file_df.iloc[[n_segments - 1]]['End']) # seconds
    real_audio_length = info.num_frames / info.sample_rate # seconds
    logger.debug('Audio length: ' + str(round(real_audio_length, 2)))
    logger.debug('Labels length: ' + str(round(labels_audio_length, 2)))

    # Fix temporal information
    file_df = fix_time_reference(
        file_df,
        vad_file_df,
        real_audio_length,
        n_segments
        )

    n_rows = len(file_df.index)

    # Instantiate and initialize exceptions counter
    exceptions_counter = 0
    
    # Loop all segments of the audio
    #for row_index, row in file_df.iterrows():
    for row_index in range(n_rows):
        row = file_df.iloc[row_index]
        
        # Readability
        logger.debug('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # If segment type is non-speech
        if(row['Type'] ==  'Non-Speech'):
            logger.debug('Skipping non-speech segment...')
            print(row)
            new_segment_start = float(row['End'])
            continue

        else:
            # Get row data
            is_last_segment = True if (row_index + 1) == n_rows else False
            clip_start = new_segment_start if new_segment_start is not None else float(row['Start'])
            clip_end = float(row['End'])
            transcript = str(row['Transcription']).upper()
            clip_length = clip_end - clip_start

            # Prepare text: list of utterances
            transcript = prepare_text(transcript, max_words_sequence=max_words_sequence)
            list_of_splits.append(len(transcript))
            
            # Add discarded text from previous alignments
            if(discarded_transcripts):
                transcript = discarded_transcripts[::-1] + transcript
                discarded_transcripts = []

            # Get text to audio proportion
            text_length = count_text_length(transcript)

            try:
                text_to_audio_proportion = get_text_to_audio_proportion(
                    int(clip_length * info.sample_rate),
                    text_length,
                    info.sample_rate
                    )
            except:
                logger.debug('NaN values...')
                logger.debug(row)

            if not is_last_segment:
                following_row = file_df.iloc[row_index + 1]
                next_row_is_non_speech = True if following_row['Type'] == 'Non-Speech' else False
            
            # Check if recalculate time references. Reasons:
            #   1) Window gets big
            #   2) Speech segment is ending, ans there is non-aligned text 
            recalculate_time_references = False
            if (clip_length >= max_window_size) or \
                (text_to_audio_proportion > 10.0 and next_row_is_non_speech and \
                abs(float(following_row['Start']) - clip_start) > 5.0):
                recalculate_time_references = True

            # Stop execution if window too big
            if clip_length >= window_to_stop:
                break
            
            if recalculate_time_references:
                logger.debug('Recalculating time references, using last anchor as beginning...')
                file_df = fix_text_to_time_proportion(
                    file_df,
                    vad_file_df,
                    real_audio_length - clip_start,
                    get_n_aligned_rows(list_of_splits, len(file_alignments)),
                    n_segments,
                    clip_start,
                    logger
                    )
                row = file_df.iloc[row_index]
                clip_end = float(row['End'])
                clip_length = clip_end - clip_start
                logger.debug(
                    'Debug values. Clip Start: {0}, clip_end: {1}, aligned utterances: {2}, segments: {3}'\
                        .format(clip_start, clip_end, len(file_alignments), n_segments)
                    )

            try:
                # Read audio and normalize
                audio, sr = torchaudio.load(
                    audio_path,
                    frame_offset=int(clip_start * info.sample_rate),
                    num_frames=int(clip_length * info.sample_rate), 
                    channels_first=False
                )
                audio_normalized = asr_model.audio_normalizer(audio, sr)
                audio_length = audio.shape[0] # samples
            except:
                print('Start frame: {0}. Enf frame: {1}. Row: {2}'.format(clip_start, clip_end, row))
                print(file_df)

            # Get audio to text proportion
            text_to_audio_proportion = get_text_to_audio_proportion(audio_length, text_length, sr)
            logger.debug('Text to audio proportion: ' + str(text_to_audio_proportion))

            # Make sure that the audio and text proportions are accurate
            if not is_last_segment:
                
                # Non-valid audio length
                if not(audio_length > 0):
                    logger.debug('File {0} has non valid segment from {1} to {2}'.format(audio_path, clip_start, clip_end))
                    discarded_transcripts = transcript[::-1]
                    continue
                
                # Too much audio and low text quantity: read more text
                elif(text_to_audio_proportion < min_text_to_audio_prop):
                    logger.debug('Low quantity of text compared to audio, reading more audio and text...')
                    discarded_transcripts = transcript[::-1]
                    logger.debug('Text to be aligned: ' + str(discarded_transcripts))
                    new_segment_start = clip_start
                    # TODO: Read only more text
                    continue

                # If next segment is a non-speech segment: 
                # Text quantity too high: allow alignment discarding some text
                if text_to_audio_proportion > 10.0 and next_row_is_non_speech and \
                    abs(float(following_row['Start']) - clip_start) > 5.0:
                    logger.debug('As the speech segment is finishing, allowing alignment with a valid amount of text...')
                    transcript, discarded_transcripts = find_a_valid_text_to_audio_proportion(
                        audio_length,
                        transcript,
                        samples_to_frames_ratio
                    )

            try:
                bad_alignment = True
                repeat_alignment = True
                already_reapeating = False # control if a repetition already started
                previous_segmentation = []
                previous_absolute_end = 0.0

                # Get AM posteriors
                lpz = aligner.get_lpz(audio_normalized)
                
                while bad_alignment or repeat_alignment:
                    
                    logger.debug('Segment from {0} to {1}'.format(clip_start, clip_end))

                    # Conflate text & lpz & config as a segmentation task object
                    task = aligner.prepare_segmentation_task(
                        transcript,
                        lpz,
                        row['Sample_ID'],
                        audio_normalized.shape[0]
                        )
                    
                    # Apply CTC segmentation
                    segments = aligner.get_segments(task)
                    task.set(**segments)
                    segments = task.__str__().strip().split("\n")                
                    list_of_segments = [segment.split(" ", 5) for segment in segments]

                    for segment in list_of_segments:
                        
                        if(len(segment) != 6):
                            logger.debug('Some problem with segment: ' + str(segment))
                            continue

                        segment_transcript = segment[-1]
                        segment_start = float(segment[2])
                        segment_end = float(segment[3])
                        segment_score = float(segment[4])
                        absolute_start = clip_start + segment_start
                        absolute_end = clip_start + segment_end
                        segment_length = segment_end - segment_start
                        segment_id = "_".join([
                            audio_path.split('/')[-1].replace('.wav', ''),
                            str(absolute_start),
                            str(absolute_end)]
                            )

                        # Do not use short utterances as anchors
                        if(len(segment_transcript) < short_utterance_len):
                            segment_score += 2*threshold

                        # Check quality of last alignment
                        if(segment_score < threshold):
                            bad_alignment = True
                        else:
                            bad_alignment = False
                            new_segment_start = absolute_end

                        logger.debug('{0} | {1} | {2} | {3}'.format(
                            round(absolute_start, 3),
                            round(absolute_end, 3),
                            round(segment_score, 3),
                            segment_transcript
                            )
                        )
                        file_alignments.append(
                            [segment_id, audio_path, row['Channel'], segment_length, absolute_start, absolute_end, segment_score, segment_transcript, row['Speaker_ID'], 'RTVE2018']
                            )

                    # If there is not more audio and text to align: keep last alignment
                    if(is_last_segment):
                        bad_alignment = False
                        repeat_alignment = False
                    else:

                        # Repeat segmentation (threshold-based decision)
                        if(bad_alignment and repeat_alignment and not previous_segmentation): # check if there is improvement

                            # Impossible to iterate: read more data
                            if(not transcript[:-1]):
                                discarded_transcripts.append(transcript[-1])
                                logger.debug('Alignment of empty string cannot be done. Reading more audio to better align.')
                                bad_alignment = False
                                repeat_alignment = False
                                file_alignments = file_alignments[:len(file_alignments) - len(list_of_segments)]
                                new_segment_start = clip_start

                            # Iteration step
                            else:
                                discarded_transcripts.append(transcript[-1])
                                transcript = transcript[:-1]
                                file_alignments = file_alignments[:len(file_alignments)- len(list_of_segments)]
                                logger.debug('Misalignment detected. Repeating alignment due low score.')
                                logger.debug('Not included transcripts: ' + str(discarded_transcripts))

                        # Iterate to check if there is improvement
                        elif((not bad_alignment or previous_segmentation) and repeat_alignment):
                            
                            if(previous_segmentation):

                                previous_score = float(previous_segmentation[-2][4])
                                if(len(previous_segmentation[-2][-1]) < short_utterance_len):
                                    previous_score += 2*threshold
                                
                                # Allow punctual bad aligments if last alignment is very good:
                                if(segment_score > -1.0 and not (previous_score == segment_score)):
                                    new_segment_start = absolute_end
                                    bad_alignment = False
                                    repeat_alignment = False
                                    file_alignments = file_alignments[:len(file_alignments)-(2*len(list_of_segments) + 1)] + file_alignments[len(file_alignments) - len(list_of_segments):]
                                    logger.debug('Not repeating because last alignment is very nice. Punctual bad score could be due other reasons.')

                                # Not improved results
                                elif(previous_score >= segment_score):
                                    repeat_alignment = False
                                    bad_alignment = False
                                    new_segment_start = previous_absolute_end
                                    discarded_transcripts = discarded_transcripts[:-1]
                                    logger.debug('DEBUG: List of segments len: ' + str(len(list_of_segments)))
                                    file_alignments = file_alignments[:len(file_alignments) - len(list_of_segments)]
                                    logger.debug('Not improved results keeping previous alignment. Previous score: {0} | Current score: {1}'.format(previous_score, segment_score))
                                
                                # Results have improved
                                else:
                                    
                                    # There is no way to keep iterating
                                    if(not transcript[:-1]):

                                        # If not bad score
                                        if(not bad_alignment):
                                            new_segment_start = absolute_end
                                            logger.debug('Alignment of empty string cannot be done. Storing this last alignment ...')
                                            file_alignments = file_alignments[:len(file_alignments) - (2*len(list_of_segments) + 1)] + file_alignments[len(file_alignments) - len(list_of_segments):]
                                        else:
                                            discarded_transcripts.append(transcript[-1])
                                            logger.debug('Alignment of empty string cannot be done. Reading more audio to better align.')
                                            file_alignments = file_alignments[:len(file_alignments) - (2*len(list_of_segments) + 1)]
                                            new_segment_start = clip_start

                                        bad_alignment = False
                                        repeat_alignment = False

                                    # Results have improved: iterate again
                                    else:

                                        # TODO: here check bad_alignment
                                        # If bad alignment: then ignore last iteration and keep previous
                                        if bad_alignment:
                                            repeat_alignment = False
                                            bad_alignment = False
                                            new_segment_start = previous_absolute_end
                                            discarded_transcripts = discarded_transcripts[:-1]
                                            logger.debug('DEBUG: List of segments len: ' + str(len(list_of_segments)))
                                            file_alignments = file_alignments[:len(file_alignments) - len(list_of_segments)]
                                            logger.debug('Improved results but score is under the treshold. Keeping previous alignment. Previous score: {0} | Current score: {1}'.format(previous_score, segment_score))
                                        else:
                                            previous_segmentation = list_of_segments
                                            previous_absolute_end = absolute_end
                                            discarded_transcripts.append(transcript[-1])
                                            transcript = transcript[:-1]
                                            file_alignments = file_alignments[:len(file_alignments) - (2*len(list_of_segments) + 1)] + file_alignments[len(file_alignments) - len(list_of_segments):]
                                            logger.debug('Results have improved. Continuing iteration...')
                                        
                            # If it is the first repetition
                            else:

                                # Allow punctual bad aligments if last alignment is very good:
                                if(segment_score > -1.0):
                                    new_segment_start = absolute_end
                                    bad_alignment = False
                                    repeat_alignment = False
                                    logger.debug('Not repeating because last alignment is very nice. Punctual bad score could be due other reasons.')
                                
                                # When there is no possibility to continue iterating and the alignment is good:
                                elif(not transcript[:-1]):
                                    new_segment_start = absolute_end
                                    bad_alignment = False
                                    repeat_alignment = False
                                    logger.debug('Alignment of empty string cannot be done. Storing this last alignment ...')
                                else:
                                    discarded_transcripts.append(transcript[-1])
                                    transcript = transcript[:-1]
                                    previous_segmentation = list_of_segments
                                    previous_absolute_end = absolute_end
                                    logger.debug('Good results. Starting iteration...')

                            logger.debug('Not included transcripts: ' + str(discarded_transcripts))
                    
                    if len(file_alignments) > 1:
                        logger.debug('DUBUG last elements: ' + str(file_alignments[-1]))

                    # Readability: iterations
                    logger.debug('------------------------------[Alignment iteration]-------------------------------------')

                # Re-initialize exceptions counter
                exceptions_counter = 0

            except AssertionError as e:
                logger.debug(e)
                logger.debug('File {0} sequence from {1} to {2} is shorter than text: {3}'.format(audio_path, clip_start, clip_end, transcript))
                discarded_transcripts += transcript[::-1]
                
                # Check number of consecutive exceptions
                exceptions_counter += 1
                if exceptions_counter >= max_text_to_audio_prop_exec:
                    logger.debug("Number of reached the limit {0}/{1}".format(exceptions_counter, max_text_to_audio_prop_exec))
                    break
                else:
                    logger.debug("Number of exceptions {0}/{1}".format(exceptions_counter, max_text_to_audio_prop_exec))
                    continue

    return file_alignments


def main(args):

    # Columns to be used in results files
    columns = ['Sample_ID', 'Sample_Path', 'Channel', 'Audio_Length', 'Start', 'End','Segment_Score', 'Transcription', 'Speaker_ID', 'Database']

    # Load ASR model
    source_path = 'config/'
    hparams_path = 'ctc_sp_with_wav2vec.yaml'
    savedir_path = 'data/asr/ctc/savedir'
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path) 

    # Segmentation tool
    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, time_stamps="fixed", scoring_length=30)
    samples_to_frames_ratio = aligner.estimate_samples_to_frames_ratio()

    # Splits to align
    #splits = ['dev1', 'dev2', 'train']
    splits = ['train', 'dev1', 'dev2', 'test']

    for split in splits:

        # Scan reference files
        df_path = os.path.join(args.tsv, 'tsv', split + '.tsv')
        vad_segments_path = os.path.join(args.tsv, 'vad_segments', split + '_vad_segments_filtered.tsv')

        if(os.path.isfile(df_path) and os.path.isfile(vad_segments_path)):
            df = pd.read_csv(df_path, header=0, sep='\t')
            vad_df = pd.read_csv(vad_segments_path, header=0, sep='\t')
            audio_paths = df['Sample_Path'].unique()
            progress_bar = tqdm(total=len(audio_paths), desc='Alignment ' + split + ' split')

            # Loop each audio in chunks for the sake of computational load
            for audio_path in audio_paths:
                
                file_alignments = []
                
                if(os.path.isfile(os.path.join(args.dst, split, audio_path.split('/')[-1].replace('.wav', '.tsv')))):
                    print('File ' + str(os.path.join(args.dst, split, audio_path.replace('.wav', '.tsv'))) + ' already exist, skipping the alignment generation.')
                    progress_bar.update(1)
                    continue
                else:
                    # To allow parallelization
                    tsv_result_file = os.path.join(args.dst, split, audio_path.split('/')[-1].replace('.wav', '.tsv'))
                    open(tsv_result_file, 'a').close()

                    file_df = df[df['Sample_Path'] == audio_path]
                    file_df = file_df.reset_index(drop=True)
                    vad_file_df = vad_df[vad_df['Sample_Path'] == audio_path]
                    vad_file_df = vad_file_df.reset_index(drop=True)
                    file_alignments = get_file_iterative_segmentation(
                        asr_model,
                        aligner,
                        audio_path,
                        file_df,
                        vad_file_df,
                        samples_to_frames_ratio,
                        split
                        )
                    file_alignments_df = pd.DataFrame(file_alignments, columns=columns)
                    file_alignments_df.to_csv(tsv_result_file, sep='\t', index=None)
                
                progress_bar.update(1)
        
        else:
            print('{0} or {1} file does not exists, please create it.'.format(df_path, vad_segments_path))  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Iterative algorithm for RTVE2018DB")
    parser.add_argument("--tsv", help="metadata with filtered audio", default="data/RTVE2018DB/")
    parser.add_argument("--dst", help="metadata with utterance-level segmented audio", default="data/RTVE2018DB/segmented")
    args = parser.parse_args()

    # Run main
    main(args)