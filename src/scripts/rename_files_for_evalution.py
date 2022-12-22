import os
import shutil


def move_and_rename(origin, destination, suffix):

    hypothesis_files = [f for f in os.listdir(origin) if os.path.isfile(os.path.join(origin, f))]
    
    for file in hypothesis_files:

        if file.endswith(".txt"):
            src = os.path.join(origin, file)
            dst = os.path.join(destination, file.replace('.txt', suffix + '.txt'))
            shutil.copyfile(src, dst)


def main():

    # Primary
    primary_transcripts = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/transcription/test_but_segments+LM'
    destination = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/submission/TID_P-VADLM'
    suffix = '_TID_P-VADLM'
    move_and_rename(primary_transcripts, destination, suffix)

    # Contrastive 1
    contrastive_1 = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/transcription/test_but_segments'
    destination = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/submission/TID_C1-VAD'
    suffix = '_TID_C1-VAD'
    move_and_rename(contrastive_1, destination, suffix)

    # Contrastive 2
    contrastive_2 = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/transcription/test_w=10+LM'
    destination = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/submission/TID_C2-W10LM'
    suffix = '_TID_C2-W10LM'
    move_and_rename(contrastive_2, destination, suffix)

    # Contrastive 3
    contrastive_3 = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/transcription/test_w=10'
    destination = '/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2022DB/submission/TID_C3-W10'
    suffix = '_TID_C3-W10'
    move_and_rename(contrastive_3, destination, suffix)
    


if __name__ == "__main__":
    main()