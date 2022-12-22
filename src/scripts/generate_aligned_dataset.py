import os
import json
import shutil
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm


def main():

    # Filtering method
    #filtering_method = "score-normalization" # score-normalization, distribution, restrictive
    filtering_method = "restrictive"

    # Destination file to place results
    dst = '/home/cx02775/data/Albayzin-aligned/'

    # Files to concatenate
    repo_path = '/home/cx02775/speech-segmentation/'
    tsvs = [
        #RTVE2018DB
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/train_aligned.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/dev1_aligned.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/dev2_aligned.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/test_aligned.tsv",
        #RTVE2022DB:
        repo_path + "data/RTVE2022DB/segmented/3rd_pass/train_aligned.tsv"
    ]

    # Splits definition
    dev_shows = ['LA24H', 'LT24HEnt', 'Millennium', 'millennium', 'LT24HTiempo']
    dev2_shows = ['LT24HEco', 'AFI']
    train_shows = ['EC', 'LT24HTer', 'LM', 'SG', 'AV', 'DH', 'LN24H', '20H', 'CA', 'AP', 'AG']

    # As we do not known show name
    train_2022 = [
        '4695671', '5469407', '5451561', '5361980', '5250439', '5150393', '5468610', '4676778',
        '5462153', '5415147', '4642723', '5322944', '5233948', '5466550', '5445458', '5438876',
        '4670242', '5361435', '5395894', '5220540', '5463402', '4645618', '4820818', '5347099',
        '5394918', '5377297', '5473940', '5475384', '5470793', '5376629', '4606205', '4656442',
        '5358610', '5385910', '5087158', '5363144', '5461239', '5456599', '4672740', '5356818',
        '5338178', '4825139', '5292398', '5363819', '4858218', '5457697', '5231438', '5473270',
        '5348960', '5462467', '5231639', '5442906', '5387538', '4148579', '4750859', '5473871',
        '5368464', '4593218', '5300563', '5247248', '4658921', '5244663', '4791418', '5470723',
        '5345718', '5472355', '5197299', '5472042', '5359212', '5370006', '5471782', '5473042',
        '5392708', '4613933', '5465857', '5282178', '5427561', '5286299', '5392710', '5357172',
        '4621518', '5355488', '4262370', '5258928', '5372660', '5345358', '5473942', '5392679',
        '4835486', '5474762', '4689638', '5464172', '4623545', '5366597', '4608525', '5420939',
        '5213059', '5449773', '4740918', '5255302', '5178012', '5250289', '5360671', '5189686',
        '5261086', '5470735', '4635895', '5390937', '4730398', '5272498', '5469754', '5423322',
        '5268920', '5191586', '5281041', '4616819', '5373265', '5460284', '5464991', '5436602',
        '4649647', '5438959', '5288178', '5172567', '5467742', '4764743', '5468091', '4684741',
        '5388966', '5459412', '4663822', '5470443', '5369485', '5204558', '5465869', '5461852',
        '5368990', '5357995', '5315018', '5193898', '5187152', '5163758', '5261220', '5187479',
        '5236146', '5274260', '4648742', '5408187', '4628778', '5392631', '5391787', '5277098',
        '5397669', '4631462', '4666165', '5267180', '5408135', '5238264', '5396667', '5423358',
        '5360372', '5445566', '5387509', '5114419', '5475068', '5354879', '5355574', '5370523',
        '4586801', '5385212', '5131745', '5240239', '5385923', '5096073', '5263172', '5359694',
        '4692239', '5165391', '4711959', '5468960', '4652722', '4781221', '5253560', '5381598',
        '5397639', '4820799', '4733029', '5456566', '5467023', '5463810', '5473918', '5350758',
        '5464616', '5412453', '4600518', '5341315', '5465547', '5307198', '5279300', '5234179',
        '4743959', '5274381', '5247482', '5387539', '5332338', '5152580', '5209458', '5161979',
        '5189052', '5167210', '5213158', '5395833', '5285220', '5465895', '5467325', '5366008',
        '4681818', '5456161', '5402375', '5185412', '5400802', '5159827', '5454796', '5181799',
        '5212692', '5386711', '4815258', '4805229', '5431018', '5265165', '5362503', '5298518',
        '5174478', '5276880', '4638299', '5174838', '5367409', '5406506', '5473651', '5161813',
        '4703599', '5390955', '5362621', '5461173', '5394138', '4641458', '4577758', '5451529',
        '5365434', '5252052', '5359153', '5356161', '5176422', '5462979', '5460671', '5475594',
        '5390205', '5163554', '5263378']

    train_shows = train_shows + train_2022

    list_of_dataframes = []

    for tsv in tsvs:
    
        if(os.path.isfile(tsv)):
            
            # Get partition name
            partition_name = tsv.split('/')[-1].replace('_aligned.tsv', '')
            partition_df = pd.read_csv(tsv, header=0, sep='\t')
            
            # Add original split name 
            partition_df['Original_Split'] = partition_name

            # Add a temporal column with clip path
            partition_df['Clip_Path'] = os.path.join(tsv.split('segmented')[0],  'segmented_clips', tsv.split('segmented')[1].split('/')[1])
       
            # Append to list
            list_of_dataframes.append(partition_df)
            
        else:
            print('file does not exist')
            break

    aligned_database_df = pd.concat(list_of_dataframes, ignore_index=True)
    aligned_database_df.to_csv(os.path.join(dst, 'tsv','albayzin_aligned.tsv'), index=None, sep='\t')
    
    #####################
    # Filtering methods #
    #####################

    # Method-1: score length normalization & th=-1.5
    if filtering_method == "score-normalization":
        subname = "sn"
        threshold = -1.5
        U = 8.0 # heuristic mean utterance length from dev1, dev2 and test
        aligned_database_df['Segment_Score'] = (aligned_database_df['Segment_Score'] + np.log(aligned_database_df['Audio_Length']/U))
    
    # Method-2: threshold = Mean-2SD
    elif filtering_method == "distribution":
        subname = "ds"
        mean_log = aligned_database_df['Segment_Score'].mean()
        std_log = aligned_database_df['Segment_Score'].std()
        threshold = round(mean_log/(2 + std_log), 2)
    
    # Method-3: threshold = -1
    elif filtering_method == "restrictive":
        subname = "rs"
        threshold = -1.0

    else:
        print("One of the filtering methods must be selected.")
        return
    
    # Store complete dataset
    filtered_database_df = aligned_database_df[aligned_database_df['Segment_Score'] > threshold]
    filtered_database_df.to_csv(os.path.join(dst, 'tsv', subname + '_albayzin_aligned.tsv'), index=None, sep='\t')

    #####################
    #### Check audio ####
    #####################

    # Check all audio files are available and place them in <dst>/clips/ path
    # Assumption of generated clips in repo_path/segmented_clips/<split_name>/
    dst_clips = os.path.join(dst, 'clips')

    progress_bar = tqdm(
        total=len(filtered_database_df.index),
        desc='copying audio clips of database',
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )

    for index, row in filtered_database_df.iterrows():
        src_path = os.path.join(row['Clip_Path'], row['Sample_ID'] + '.wav')
        dst_path = os.path.join(dst_clips, row['Sample_ID'] + '.wav')

        # Check if already exist file:
        if not os.path.isfile(dst_path):
            shutil.copyfile(src_path, dst_path)

        progress_bar.update(1)

    # Drop unneded column
    filtered_database_df.drop(['Clip_Path'], axis=1, inplace=True)

    # Store splits
    train_df = filtered_database_df[filtered_database_df['Sample_ID'].str.contains("|".join(train_shows))]
    train_df.to_csv(os.path.join(dst, 'tsv', subname + '_train.tsv'), index=None, sep='\t')
    dev_df = filtered_database_df[filtered_database_df['Sample_ID'].str.contains("|".join(dev_shows))]
    dev_df.to_csv(os.path.join(dst, 'tsv', subname + '_dev.tsv'), index=None, sep='\t')
    dev2_df = filtered_database_df[filtered_database_df['Sample_ID'].str.contains("|".join(dev2_shows))]
    dev2_df.to_csv(os.path.join(dst, 'tsv', subname + '_dev2.tsv'), index=None, sep='\t')

    # Generate information json
    total_samples_aligned = len(aligned_database_df.index)
    total_aligned_hours = aligned_database_df['Audio_Length'].sum() / 3600.0
    mean_aligned_segment_length = aligned_database_df['Audio_Length'].mean()

    total_samples_filtered = len(filtered_database_df.index)
    total_filtered_hours = filtered_database_df['Audio_Length'].sum() / 3600.0
    mean_filtered_segment_length = filtered_database_df['Audio_Length'].mean()

    train_samples = len(train_df.reset_index().index)
    train_hours = train_df['Audio_Length'].sum() / 3600.0

    dev_samples = len(dev_df.reset_index().index)
    dev_hours = dev_df['Audio_Length'].sum() / 3600.0

    dev2_samples = len(dev2_df.reset_index().index)
    dev2_hours = dev2_df['Audio_Length'].sum() / 3600.0

    data = {
        "aligned": {
            "samples": total_samples_aligned,
            "hours": total_aligned_hours,
            "mean_segment_length(s)": mean_aligned_segment_length
        },
        "filtered": {
            "samples": total_samples_filtered,
            "hours": total_filtered_hours,
            "mean_segment_length(s)": mean_filtered_segment_length
        },
        "train": {
            "samples": train_samples,
            "hours": train_hours,
        },
        "dev": {
            "samples": dev_samples,
            "hours": dev_hours
        },
        "dev2": {
            "samples": dev2_samples,
            "hours": dev2_hours
        }
    }

    with open(os.path.join(dst, 'tsv', subname + '_info.json') , 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    # Run main
    main()