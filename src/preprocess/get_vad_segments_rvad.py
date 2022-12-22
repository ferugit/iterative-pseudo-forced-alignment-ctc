import sys
sys.path.insert(1, sys.path[0].replace('/' + sys.path[0].split('/')[-1], ''))

import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.signal import lfilter

import utils.rvad_utils as rvad_utils


def rvad_segments(WAV_PATH):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    finwav = WAV_PATH

    fs, data = rvad_utils.speech_wave(finwav)
    ft, flen, fsh10, nfr10 = rvad_utils.sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(nfr10)
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    pvblk = rvad_utils.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770,   -0.9770])
    a = np.array([1.0000,   -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    noise_samp, noise_seg, n_noise_samp = rvad_utils.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]),  int(noise_samp[j, 1]) + 1)] = 0

    vad_seg = rvad_utils.snre_vad(
        fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)

    frame_array = []
    
    # Segun el paper son 10ms de frame shift
    for i, frame in enumerate(vad_seg):
        
        if frame == 0:
            frame_array.append((i*10, False))
        else:
            frame_array.append((i*10, True))

    segments = rvad_utils.frame_array_to_segments(frame_array)
    
    return segments


def main(args):
    original_tsv = pd.read_csv(args.src, sep="\t")
    result_tsv = pd.DataFrame(
        columns=["Sample_Path", "Audio_Length", "Start", "End", "Segment_Length"])

    for index, row in tqdm(original_tsv.iterrows(), total=len(original_tsv), desc="Processing with rVAD"):
        segments = rvad_segments(row["Sample_Path"])
        
        for segment in segments:
            # Add new row to result_tsv
            result_tsv.loc[len(result_tsv)] = [
                row["Sample_Path"],
                row["Audio_Length"],
                segment[0],
                segment[1],
                segment[1] - segment[0]
            ]

    result_tsv.to_csv(os.path.join(args.dst, args.src.split(
        '/')[-1].replace('.tsv', '_rvad_segments.tsv')), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script generate VAD segments  with rVAD method"
        )

    parser.add_argument(
        "--src", help="tsv with to get speech segments", default=""
        )
    parser.add_argument(
        "--dst", help="path to place resultant tsv", default=""
        )
    args = parser.parse_args()

    main(args)
