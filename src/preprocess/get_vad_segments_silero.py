import os
import argparse
from tqdm import tqdm

import torch
import torchaudio
import pandas as pd


def load_silero(model_dir: str = "models/silero"):
    """Load Silero VAD model, caching weights under model_dir."""
    os.makedirs(model_dir, exist_ok=True)
    torch.hub.set_dir(model_dir)
    from silero_vad import load_silero_vad
    return load_silero_vad()


def get_speech_segments(model, audio_path: str):
    """
    Return speech timestamps (seconds) for a single audio file.

    Returns
    -------
    list[dict]
        Each dict has 'start' and 'end' keys in seconds.
    """
    from silero_vad import get_speech_timestamps

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)

    return get_speech_timestamps(wav, model, return_seconds=True)


def main(args):
    if not (os.path.isfile(args.src) and os.path.isdir(args.dst)):
        print("Source file or destination directory does not exist.")
        return

    df = pd.read_csv(args.src, header=0, sep="\t")
    audio_paths = df["Sample_Path"].unique()
    progress_bar = tqdm(total=len(audio_paths), desc="Processing with VAD")

    model = load_silero(model_dir=args.model_dir)

    vad_rows = []
    for audio_path in audio_paths:
        info = torchaudio.info(audio_path)
        audio_len = info.num_frames / info.sample_rate

        segments = get_speech_segments(model, audio_path)
        for seg in segments:
            vad_rows.append([audio_path, audio_len, seg["start"], seg["end"]])

        progress_bar.update(1)

    out_df = pd.DataFrame(vad_rows, columns=["Sample_Path", "Audio_Length", "Start", "End"])
    out_df["Segment_Length"] = out_df["End"] - out_df["Start"]
    out_name = os.path.basename(args.src).replace(".tsv", "_vad_segments.tsv")
    out_df.to_csv(os.path.join(args.dst, out_name), sep="\t", index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VAD segments using Silero VAD")
    parser.add_argument("--src", help="input TSV with Sample_Path column", default="")
    parser.add_argument("--dst", help="directory for output TSV", default="")
    parser.add_argument("--model_dir", help="directory to cache model weights", default="models/silero")
    args = parser.parse_args()
    main(args)
