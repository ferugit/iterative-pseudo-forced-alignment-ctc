import pandas as pd


def get_audio_overlap(start_1, end_1, start_2, end_2):
        '''
        Get overlap time in seconds of two sound events
        '''
        
        overlap = 0.0

        if(start_1 > start_2 and end_2 > start_1):
            if(end_2 < end_1):
                overlap = end_2 - start_1
            elif(end_2 > end_1):
                overlap = end_1 - start_1
        elif(start_1 < start_2 and start_2 < end_1):
            if(end_2 < end_1):
                overlap = end_2 - start_2
            elif(end_2 > end_1):
                overlap = end_1 - start_2

        return overlap


def get_bounds_deviation_seconds(start_1, end_1, start_2, end_2):
        '''
        Get overlap time in seconds of two sound events
        '''
        
        left_deviation = abs(start_1 - start_2)
        right_deviation =  abs(end_1 - end_2)
        return left_deviation, right_deviation


def main():

    ctc_segmentation = 'data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips/segmented_ctc.tsv'
    seq2seq_segmentation = 'data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips/segmented_seq2seq.tsv'

    # Assumend as ground truth
    handmade_segmentation = 'data/commonvoice/cv-corpus-7.0-2021-07-21/es/segmented_clips/segmented_hand.tsv'

    ctc_df = pd.read_csv(ctc_segmentation, header=0, sep='\t')
    seq2seq_df = pd.read_csv(seq2seq_segmentation, header=0, sep='\t')
    handmade_df = pd.read_csv(handmade_segmentation, header=0, sep='\t')

    ctc_deviations = []
    ctc_overlaps = []
    ctc_left_deviations = []
    ctc_right_deviations = []
    seq2seq_deviations = []
    seq2seq_overlaps = []
    seq2seq_left_deviations = []
    seq2seq_right_deviations = []

    for _, row in handmade_df.iterrows():

        audio_file = row['path']
        start = row['start']
        end = row['end']
        word_length = end-start
        print('Audio file:' , audio_file.split('/')[-1])
        
        ctc_start = ctc_df.set_index('path').at[audio_file, 'start']
        ctc_end = ctc_df.set_index('path').at[audio_file, 'end']
        if(isinstance(ctc_start, pd.Series)):
            ctc_start = ctc_start[0]
        if(isinstance(ctc_end, pd.Series)):
            ctc_end = ctc_end[0]
        ctc_overlap = get_audio_overlap(start_1=start, end_1=end, start_2=ctc_start, end_2=ctc_end)
        ctc_left_deviation, ctc_right_deviation = get_bounds_deviation_seconds(start_1=start, end_1=end, start_2=ctc_start, end_2=ctc_end)
        ctc_deviation = ctc_left_deviation + ctc_right_deviation
        ctc_deviations.append(ctc_deviation)
        ctc_overlaps.append(100*(ctc_overlap/word_length))
        ctc_left_deviations.append(ctc_left_deviation)
        ctc_right_deviations.append(ctc_right_deviation)
        print('\tCTC')
        print('\t\tOverlap: ' , round(100*(ctc_overlap/word_length), 2), '%')
        print('\t\tBoundaries deviation:', round(ctc_deviation, 2), 'seconds')
        print('\t\tLeft bound deviation:', round(ctc_left_deviation, 2), 'seconds')
        print('\t\tRight bound deviation:', round(ctc_right_deviation, 2), 'seconds')

        seq2seq_start = seq2seq_df.set_index('path').at[audio_file, 'start']
        seq2seq_end = seq2seq_df.set_index('path').at[audio_file, 'end']
        seq2seq_overlap = get_audio_overlap(start_1=start, end_1=end, start_2=seq2seq_start, end_2=seq2seq_end)
        seq2seq_left_deviation, seq2seq_right_deviation = get_bounds_deviation_seconds(start_1=start, end_1=end, start_2=seq2seq_start, end_2=seq2seq_end)
        seq2seq_deviation = seq2seq_left_deviation + seq2seq_right_deviation
        seq2seq_deviations.append(seq2seq_deviation)
        seq2seq_overlaps.append(100*(seq2seq_overlap/word_length))
        seq2seq_left_deviations.append(seq2seq_left_deviation)
        seq2seq_right_deviations.append(seq2seq_right_deviation)
        print('\tseq2seq')
        print('\t\tOverlap:',  round(100*(seq2seq_overlap/word_length), 2), '%')
        print('\t\tBoundaries deviation:', round(seq2seq_deviation, 2), 'seconds')
        print('\t\tLeft bound deviation:', round(seq2seq_left_deviation, 2), 'seconds')
        print('\t\tRight bound deviation:', round(seq2seq_right_deviation, 2), 'seconds')


    # Average results
    print('-------------------------------------------------')
    print('CTC mean overlap:', round(sum(ctc_overlaps)/len(ctc_overlaps), 2), '%')
    print('CTC boundaries mean deviation:', round(sum(ctc_deviations)/len(ctc_deviations), 2), 'seconds')
    print('CTC left bound mean deviation:', round(sum(ctc_left_deviations)/len(ctc_deviations), 2), 'seconds')
    print('CTC right bound mean deviation:', round(sum(ctc_right_deviations)/len(ctc_deviations), 2), 'seconds')
    print('')
    print('seq2seq mean overlap:', round(sum(seq2seq_overlaps)/len(seq2seq_overlaps), 2), '%')
    print('seq2seq boundaries mean deviation:', round(sum(seq2seq_deviations)/len(seq2seq_deviations), 2), 'seconds')
    print('seq2seq left bound mean deviation:', round(sum(seq2seq_left_deviations)/len(seq2seq_deviations), 2), 'seconds')
    print('seq2seq right bound mean deviation:', round(sum(seq2seq_right_deviations)/len(seq2seq_deviations), 2), 'seconds')


if __name__ == '__main__':
    main()