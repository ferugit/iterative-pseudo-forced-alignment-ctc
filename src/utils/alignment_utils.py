
import os
import sys
import logging
import pandas as pd

from utils import text_utils


def my_custom_logger(logs_path, logger_name, level=logging.DEBUG):
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
    file_handler = logging.FileHandler(os.path.join(logs_path, logger_name + '.log'), mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def prepare_text(transcript, max_words_sequence=None, min_words_sequence=None):
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
    if max_words_sequence or min_words_sequence:

        if max_words_sequence:
            if(len(transcript.split(' ')) > max_words_sequence):
                transcript = text_utils.split_long_transcript(
                    transcript,
                    max_words_sequence=max_words_sequence
                    )
            else:
                transcript = [transcript] # transform to list

        if min_words_sequence:
            raise Exception("Min word sequence not implemented")
        
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
    
    return file_df


def find_a_valid_text_to_audio_proportion(audio_length, transcript, samples_to_frames_ratio):
    """
    Given a high quantity of text compared with the audio length.
    Find a valid text to audio proportion to calculate do the
    alignment.
    """
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

    return file_df


def remove_artefacts(df, length):
    df["Text_Length"] = df['Transcription'].apply(lambda x: len(x))
    df.loc[df['Text_Length'] < length, 'Segment_Score'] = df['Segment_Score'] + 4.0
    df.drop(['Text_Length'], axis=1, inplace=True)
    return df