import re
import string

from num2words import num2words


def convert_hours_to_words(transcript):
    """
    Detect hours format and convert them from numbers to hours
    """
    hours = [s for s in transcript.split() if re.search("^(2[0-3]|[0-1]?[0-9]):([0-5][0-9])$", s)]
    
    for item in hours:        
        hour, minutes = item.split(':')
        hour = int(hour)
        minutes = int(minutes)
        
        if(hour > 12):
            hour -= 12
        elif(hour == 0):
            hour = 12

        hour_str = num2words(hour, lang='es')

        if(minutes == 0):
            hour_in_words = hour_str + " en punto"
        elif(minutes == 30):
            hour_in_words = hour_str + " y media"
        elif(minutes == 30):
            hour_in_words = hour_str + " y cuarto"
        else: 
            minutes_str = num2words(minutes, lang='es')
            hour_in_words = hour_str + " y " + minutes_str
        
        transcript = transcript.replace(item, hour_in_words)

    return transcript

def convert_numbers_to_words(transcript):
    numbers = [int(s) for s in transcript.split() if s.isdigit()]

    if len(numbers) > 0:
        for number in numbers:
            transcript = transcript.replace(str(number), num2words(number, lang='es'))

    return transcript


def normalize_transcript(transcript):
    # Remove HTML colors
    normalized_transcript = re.sub(r"<font color=\"#[0-9a-fA-F]{6}\">", "", transcript)
    normalized_transcript = re.sub(r"</font>", "", normalized_transcript)

    # Remove line breaks
    normalized_transcript = normalized_transcript.replace('\n', ' ')

    # Convert hours to words
    _ = convert_hours_to_words(normalized_transcript)

    # Remove punctuation signs.
    normalized_transcript = normalized_transcript.translate(str.maketrans('', '', string.punctuation))
    
    # Format to lower case.
    normalized_transcript = normalized_transcript.lower()
    
    # Remove exclamation and question marks.
    normalized_transcript = normalized_transcript.replace('!', '').replace('¡', '').replace('?', '').replace('¿', '')
    
    # Remove unnecessary whitespaces.
    normalized_transcript = normalized_transcript.replace('   ', ' ').replace('  ', ' ')
    
    # Convert numbers to words.
    normalized_transcript = convert_numbers_to_words(normalized_transcript)

    # FIXME: replace M, KM, M2 with the transcription 
    # FIXME: some numbers are beeing keeped in the transcript

    return normalized_transcript


def split_long_transcript(transcript, max_words_sequence=24):
    """
    split a long character sequence in chunks of a given size
    """
    word_list = transcript.split(' ')
    transcript_len = len(word_list)
    n_chunks = int(transcript_len / max_words_sequence)
    transcript_list = []
    for i in range(n_chunks + 1):
        chunk = ' '.join(
            word_list[int(i*max_words_sequence):int((i+1)*max_words_sequence)]
            ).strip()
        if chunk:
            transcript_list.append(chunk)
    return transcript_list
    