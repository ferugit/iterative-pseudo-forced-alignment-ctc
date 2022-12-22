import os
import pandas as pd

import torchaudio


def main():

    # Read only RTVE2020DB segments
    #segments_file = '/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/rtve2020_test.v0.vadv0'
    database = 'RTVE2020DB'
    segments_file = '/disks/md1-8T/users/cx02275/speech-segmentation/data/but/final_vad_segments/rtve2020_test.v0_eval2022.vadv0'
    audio_path = '/disks/md1-8T/users/cx02275/data/Albayzin_2022/' + database + '/test/audio_16kHz/S2T'

    segments_list = []
    previous_show = ""

    with open(segments_file) as fp:
        lines = fp.readlines()
        
        for line in lines:

            if "rtve2020" in line and database == "RTVE2022DB":
                continue

            if "rtve2022" in line and database == "RTVE2020DB":
                continue

            line = line.split()

            uttid  = line[0]
            recording = line[1]
            start = float(line[2])
            end = float(line[3])
            
            show = recording.split("_")[1].lstrip("0")
            sample_path = os.path.join(audio_path, show + ".wav")

            if "IM-271F8DG-C0026S01" in sample_path:
                sample_path = sample_path.replace("IM-271F8DG-C0026S01-", "IM-271F8DG_C0026S01-")

            if show != previous_show:
                audio_metadata = torchaudio.info(sample_path)
                full_audio_length = audio_metadata.num_frames/audio_metadata.sample_rate

            segments_list.append([sample_path, full_audio_length, start, end, end-start])

    columns = ["Sample_Path", "Audio_Length", "Start",	"End",	"Segment_Length"]
    vad_segments_df = pd.DataFrame(segments_list, columns=columns)
    vad_segments_df.to_csv("/disks/md1-8T/users/cx02275/speech-segmentation/data/" + database + "/vad_segments/test_vad_segments_but.tsv", index=None, sep="\t")


if __name__ == "__main__":
    main()