import os
import json
import argparse

sclite_path = "/disks/md1-8T/users/cx02275/SCTK/src/sclite/sclite"
data_path = "/disks/md1-8T/users/cx02275/speech-segmentation/data/"

def main(args):

    if os.path.isdir(args.src):

        # Gille check
        if args.src == 'Guille':
            ref_folder = "/disks/md1-8T/users/b.gcr/git/speech-asr/wav2letter/recipes/models/albayzin2020_baseline/results/albayzin2020_test_ref/stm"
            results_filepath = "/disks/md1-8T/users/cx02275/speech-segmentation/data/RTVE2018DB/transcripts/test_guille"

        # RTVE2018DB
        elif 'RTVE2018DB' in args.src:
            ref_folder = "/disks/md1-8T/users/cx02275/data/Albayzin_2022/RTVE2018DB/test/references/stm"
            results_filepath = args.src

        # RTVE2020DB
        elif 'RTVE2020DB' in args.src:
            ref_folder = "/disks/md1-8T/users/cx02275/data/Albayzin_2022/RTVE2020DB/test/references/S2T/stm"
            results_filepath = args.src
        
        output_filepath = os.path.join(results_filepath, 'output')

        if not os.path.isdir(output_filepath):
            os.mkdir(output_filepath)

        file_names = []
        for root, directories, files in os.walk(results_filepath):
            for file in files:
                if file.endswith(".txt") and not file.startswith("."):
                    file_names.append(file)

        substitutions = 0
        deletions = 0
        insertions = 0
        words = 0

        print(file_names)

        for file_name in file_names:
            folder_path = os.path.join(output_filepath, file_name.replace('.txt', ''))
            
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                    
            #code = file_name.split('_')[0].replace('.txt', '')
            code = file_name.replace('.txt', '')
            
            # Fix for RTVE2018DB
            code = "LM-20170103-MD" if 'LM-20170103' in code else code

            # Fix for RTVE2020DB: audio with underscore and reference with dash
            code = code.replace('_', '-') if 'IM-271F8DG' in code else code

            sclite_command = sclite_path + " -r " + ref_folder + "/" + code + ".stm stm -h " + results_filepath + "/" + file_name + " txt -O " + folder_path + " -o all sgml"
            print(sclite_command)
            os.system(sclite_command)
            
            # Read sys file with results.
            with open(os.path.join(folder_path, file_name + '.raw'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "#_0" in line:
                        line = " ".join(line.split())
                        info = line.split('|')
                        word_number = int(info[2].lstrip().rstrip().split(' ')[1])
                        scores = info[3].lstrip().rstrip().split(' ')

                        subs_number = int(scores[1])
                        del_number = int(scores[2])
                        ins_number = int(scores[3])

                        substitutions += subs_number
                        deletions += del_number
                        insertions += ins_number
                        words += word_number


        global_wer = 100*(substitutions + deletions + insertions)/words

        summary = {
            "Substitutions": substitutions,
            "Deletions": deletions,
            "Insertions": insertions,
            "words": words,
            "WER": global_wer,
            "reference_folder": ref_folder,
            "transcripts_folder": results_filepath
        }

        result_path = os.path.join(results_filepath, 'albayzin_wer.json')
        with open(result_path, 'w') as outfile:
            json.dump(summary, outfile, ensure_ascii=False, indent=4)

    else:
        print("Source directory does not exist")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to get the WER by program")
    parser.add_argument("--src", help="transcripts path", default="")
    args = parser.parse_args()
    main(args)