import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse
from tqdm import tqdm

from transformers import pipeline
import utils.text_utils as text_utils


def main(args):

    if(os.path.isdir(args.hypothesis_path) and os.path.isdir(args.dst)):
        
        hypothesis_files = [f for f in os.listdir(args.hypothesis_path) if os.path.isfile(os.path.join(args.hypothesis_path, f))]

        translator = pipeline("translation", model="data/corrector")

        progress_bar = tqdm(total=len(hypothesis_files), desc='hypothesis correction')
        
        for file in hypothesis_files:

            if file.endswith(".txt"):
                
                with open(os.path.join(args.hypothesis_path, file)) as f:
                    lines = f.readlines()
                    sentences = text_utils.split_long_transcript(lines[0])
                    corrected_sentences = []

                    for sentence in sentences:
                        #print(translator(sentence)['translation_text'])
                        corrected_sentences.append(translator(sentence)[0]['translation_text'])
                        
                    corrected_file = " ".join(corrected_sentences).strip()

                    dst_file = open(os.path.join(args.dst, file), "w")
                    dst_file.write(corrected_file)
                    dst_file.close()

            progress_bar.update(1)

    else:
        print("Invalid arguments.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to correct hypothesis using a transformer")
    parser.add_argument("--hypothesis_path", help="path where asr hypothesis are stored", default="")
    parser.add_argument("--dst", help="path where asr hypothesis corrected by transformer", default="")
    args = parser.parse_args()

    main(args)