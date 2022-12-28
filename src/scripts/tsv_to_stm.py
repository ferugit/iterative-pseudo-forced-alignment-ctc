import argparse

import pandas as pd



def main(args):

    #


    pass




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate clips from tsv")
    parser.add_argument('--src', help="source tsv file", default="")
    parser.add_argument("--dst", help="destination directory of cropped audios", default="")
    args = parser.parse_args()

    main(args)