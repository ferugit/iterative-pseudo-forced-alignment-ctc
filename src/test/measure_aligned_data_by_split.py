import os
import pandas as pd


def main():
    # Files to measure
    repo_path = '/home/cx02775/speech-segmentation/'
    """
    tsvs = [
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/train_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/dev1_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/dev2_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/3rd_pass/test_aligned_filtered.tsv",
        #RTVE2022DB:
        repo_path + "data/RTVE2022DB/segmented/3rd_pass/train_aligned_filtered.tsv"
    ]
    """
    tsvs = [
        #RTVE2018DB
        repo_path + "data/RTVE2018DB/segmented/train/train_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/dev1_2nd_pass/dev1_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/dev2_2nd_pass/dev2_aligned_filtered.tsv",
        repo_path + "data/RTVE2018DB/segmented/test_2nd_pass/test_aligned_filtered.tsv",
        #RTVE2022DB:
        repo_path + "data/RTVE2022DB/segmented/train/train_aligned_filtered.tsv",
    ]

    total_hours = 0

    for tsv in tsvs:
    
        if(os.path.isfile(tsv)):
            
            # Get partition name
            partition_name = tsv.split('/')[-1].replace('_aligned.tsv', '').replace('_aligned_filtered.tsv', '')
            partition_df = pd.read_csv(tsv, header=0, sep='\t')

            # Get database
            if "RTVE2018DB" in tsv:
                database = 'RTVE2018DB'
            else:
                database = 'RTVE2022DB'
            
            hours = (partition_df['Audio_Length'].sum() / 3600)
            total_hours += hours

            print(f"The {database} - {partition_name} has {hours} hours of audio.")
            
        else:
            print('file does not exist')
            break

    print(f"The total amount of hours is: {total_hours}")

if __name__ == '__main__':
    main()