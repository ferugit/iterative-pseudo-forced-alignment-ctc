import os
import json
import argparse


def main(args):
    if os.path.isdir(args.src):

        json_filename = os.path.join(os.path.dirname(args.src), 'albayzin_wer.json')
        json_results = json.load(open(json_filename, 'r'))
        directories = os.listdir(args.src)
        programs_list =  map(lambda x:  x.split('-')[0], directories)
        programs = dict.fromkeys(programs_list)
        
        for k in programs.keys():
            programs[k] = {}
            programs[k]['substitutions'] = 0
            programs[k]['deletions'] = 0
            programs[k]['insertions'] = 0
            programs[k]['words'] = 0

        for dir in directories:
            program_name = dir.split('-')[0]
            full_path = os.path.join(args.src, dir, dir + '.txt.raw')
            print(program_name)
            
            with open(full_path, 'r') as f:
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

                        programs[program_name]['substitutions'] += subs_number
                        programs[program_name]['deletions'] += del_number
                        programs[program_name]['insertions'] += ins_number
                        programs[program_name]['words'] += word_number

        for k in programs.keys():
            programs[k]['WER'] = 100*(programs[k]['substitutions'] + programs[k]['deletions'] + programs[k]['insertions'])/programs[k]['words']
        
        json_results['programs'] = programs
        
        with open(json_filename , 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)

    else:
        print("Source directory does not exist")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to get the WER by program")
    parser.add_argument("--src", help="resultant transcripts", default="data/RTVE2018DB/transcripts/38_tokenizer/test/output")
    args = parser.parse_args()
    main(args)