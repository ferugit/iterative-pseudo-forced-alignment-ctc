import os
import argparse
from tqdm import tqdm

import torch
import torchaudio
from speechbrain.pretrained import VAD

import pandas as pd


def get_speech_prob_file(
        vad,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
    ):
        """Outputs the frame-level speech probability of the input audio file
        using the neural model specified in the hparam file. To make this code
        both parallelizable and scalable to long sequences, it uses a
        double-windowing approach.  First, we sequentially read non-overlapping
        large chunks of the input signal.  We then split the large chunks into
        smaller chunks and we process them in parallel.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size:
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            True, creates overlapped small chunks. The probabilities of the
            overlapped chunks are combined using hamming windows.

        Returns
        -------
        prob_vad: torch.tensor
            Tensor containing the frame-level speech probabilities for the
            input audio file.
        """
        # Getting the total size of the input file
        sample_rate, audio_len = vad._get_audio_info(audio_file)

        if sample_rate != vad.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the length (in samples) of the large and small chunks
        long_chunk_len = int(sample_rate * large_chunk_size)
        small_chunk_len = int(sample_rate * small_chunk_size)

        # Setting the step size of the small chunk (50% overapping windows are supported)
        small_chunk_step = small_chunk_size
        if overlap_small_chunk:
            small_chunk_step = small_chunk_size / 2

        # Computing the length (in sample) of the small_chunk step size
        small_chunk_len_step = int(sample_rate * small_chunk_step)

        # Loop over big chunks
        prob_chunks = []
        last_chunk = False
        begin_sample = 0
        while True:

            # Reading the big chunk
            large_chunk, fs = torchaudio.load(
                audio_file,
                frame_offset=begin_sample,
                num_frames=long_chunk_len, 
                channels_first=False,
            )
            large_chunk = vad.audio_normalizer(large_chunk, sample_rate)
            large_chunk = large_chunk.to(vad.device)
            large_chunk = large_chunk.unsqueeze(0)

            # Manage padding of the last small chunk
            if last_chunk or large_chunk.shape[-1] < small_chunk_len:
                padding = torch.zeros(
                    1, small_chunk_len, device=large_chunk.device
                )
                large_chunk = torch.cat([large_chunk, padding], dim=1)

            # Splitting the big chunk into smaller (overlapped) ones
            small_chunks = torch.nn.functional.unfold(
                large_chunk.unsqueeze(1).unsqueeze(2),
                kernel_size=(1, small_chunk_len),
                stride=(1, small_chunk_len_step),
            )
            small_chunks = small_chunks.squeeze(0).transpose(0, 1)

            # Getting (in parallel) the frame-level speech probabilities
            small_chunks_prob = vad.get_speech_prob_chunk(small_chunks)
            small_chunks_prob = small_chunks_prob[:, :-1, :]

            # Manage overlapping chunks
            if overlap_small_chunk:
                small_chunks_prob = vad._manage_overlapped_chunks(
                    small_chunks_prob
                )

            # Prepare for folding
            small_chunks_prob = small_chunks_prob.permute(2, 1, 0)

            # Computing lengths in samples
            out_len = int(
                large_chunk.shape[-1] / (sample_rate * vad.time_resolution)
            )
            kernel_len = int(small_chunk_size / vad.time_resolution)
            step_len = int(small_chunk_step / vad.time_resolution)

            # Folding the frame-level predictions
            small_chunks_prob = torch.nn.functional.fold(
                small_chunks_prob,
                output_size=(1, out_len),
                kernel_size=(1, kernel_len),
                stride=(1, step_len),
            )

            # Appending the frame-level speech probabilities of the large chunk
            small_chunks_prob = small_chunks_prob.squeeze(1).transpose(-1, -2)
            prob_chunks.append(small_chunks_prob)

            # Check stop condition
            if last_chunk:
                break

            # Update counter to process the next big chunk
            begin_sample = begin_sample + long_chunk_len

            # Check if the current chunk is the last one
            if begin_sample + long_chunk_len > audio_len:
                last_chunk = True

        # Converting the list to a tensor
        prob_vad = torch.cat(prob_chunks, dim=1)
        last_elem = int(audio_len / (vad.time_resolution * sample_rate))
        prob_vad = prob_vad[:, 0:last_elem, :]

        return prob_vad


def get_speech_segments_from_probs(
        vad,
        audio_file,
        prob_chunks,
        apply_energy_VAD=False,
        double_check=True,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.5,
        en_deactivation_th=0.0,
        speech_th=0.50,
    ):
        """Detects speech segments within the input file. The input signal can
        be both a short or a long recording. The function computes the
        posterior probabilities on large chunks (e.g, 30 sec), that are read
        sequentially (to avoid storing big signals in memory).
        Each large chunk is, in turn, split into smaller chunks (e.g, 10 seconds)
        that are processed in parallel. The pipeline for detecting the speech
        segments is the following:
            1- Compute posteriors probabilities at the frame level.
            2- Apply a threshold on the posterior probability.
            3- Derive candidate speech segments on top of that.
            4- Apply energy VAD within each candidate segment (optional).
            5- Merge segments that are too close.
            6- Remove segments that are too short.
            7- Double check speech segments (optional).


        Arguments
        ---------
        audio_file : str
            Path to audio file.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            If True, it creates overlapped small chunks (with 50% overal).
            The probabilities of the overlapped chunks are combined using
            hamming windows.
        apply_energy_VAD: bool
            If True, a energy-based VAD is used on the detected speech segments.
            The neural network VAD often creates longer segments and tends to
            merge close segments together. The energy VAD post-processes can be
            useful for having a fine-grained voice activity detection.
            The energy thresholds is  managed by activation_th and
            deactivation_th (see below).
        double_check: bool
            If True, double checkis (using the neural VAD) that the candidate
            speech segments actually contain speech. A threshold on the mean
            posterior probabilities provided by the neural network is applied
            based on the speech_th parameter (see below).
        activation_th:  float
            Threshold of the neural posteriors above which starting a speech segment.
        deactivation_th: float
            Threshold of the neural posteriors below which ending a speech segment.
        en_activation_th: float
            A new speech segment is started it the energy is above activation_th.
            This is active only if apply_energy_VAD is True.
        en_deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
            This is active only if apply_energy_VAD is True.
        speech_th: float
            Threshold on the mean posterior probability within the candidate
            speech segment. Below that threshold, the segment is re-assigned to
            a non-speech region. This is active only if double_check is True.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.

        Returns
        -------
        boundaries: torch.tensor
            Tensor containing the start second of speech segments in even
            positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """
        # Apply a threshold to get candidate speech segments
        prob_th = vad.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Comupute the boundaries of the speech segments
        boundaries = vad.get_boundaries(prob_th, output_value="seconds")

        # Apply energy-based VAD on the detected speech segments
        #if apply_energy_VAD:
        #    boundaries = vad.energy_VAD(
        #        audio_file,
        #        boundaries,
        #        activation_th=en_activation_th,
        #        deactivation_th=en_deactivation_th,
        #    )

        # Merge short segments
        boundaries = vad.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        boundaries = vad.remove_short_segments(boundaries, len_th=len_th)

        # Double check speech segments
        if double_check:
            boundaries = vad.double_check_speech_segments(
                boundaries, audio_file, speech_th=speech_th
            )

        return boundaries


def double_check_speech_segments(
        vad, boundaries, audio_file, speech_th=0.5
    ):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.

        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = vad._get_audio_info(audio_file)

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            # Read the candidate speech segment
            segment, fs = torchaudio.load(
                audio_file, frame_offset=beg_sample, num_frames=len_seg
            )
            speech_prob = vad.get_speech_prob_chunk(segment)
            if speech_prob.mean() > speech_th:
                # Accept this a as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries


def main(args):
    
    if(os.path.isfile(args.src) and os.path.isdir(args.dst)):
        df = pd.read_csv(args.src, header=0, sep='\t')
        audio_paths = df['Sample_Path'].unique()
        progress_bar = tqdm(total=len(audio_paths), desc='Processing with VAD')

        # VAD module
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="data/vad",
        )

        vad_segments = []

        # Loop each audio in chunks for the sake of computational load
        for audio_path in audio_paths:
            info = torchaudio.info(audio_path)
            audio_len = info.num_frames / info.sample_rate 

            # Get chunks probability    
            prob_chunks = get_speech_prob_file(
                vad,
                audio_path,
                large_chunk_size=30.0,
                small_chunk_size=10.0,
                overlap_small_chunk=True
            )

            # Get speech segments
            boundaries = get_speech_segments_from_probs(
                vad,
                audio_path,
                prob_chunks,
                apply_energy_VAD=False,
                double_check=True,
                close_th=0.250,
                len_th=0.250,
                activation_th=0.4,
                deactivation_th=0.2,
                en_activation_th=0.5,
                en_deactivation_th=0.0,
                speech_th=0.50,
            )
            for segment in boundaries.tolist(): vad_segments.append([audio_path, audio_len] + segment)
            progress_bar.update(1)

        vad_segments_df = pd.DataFrame(vad_segments, columns=['Sample_Path', 'Audio_Length', 'Start', 'End'])
        vad_segments_df['Segment_Length'] = vad_segments_df['End'] - vad_segments_df['Start']
        vad_segments_df.to_csv(os.path.join(args.dst, args.src.split('/')[-1].replace('.tsv', '_vad_segments.tsv')), sep='\t', index=None)

    else:
        print('Source file or destination directory does not exist.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script generate VAD segments from RTV2018DB")

    # Alignment TSV
    parser.add_argument("--src", help="tsv with to get speech segments", default="")
    parser.add_argument("--dst", help="path to place resultant tsv", default="")
    args = parser.parse_args()

    main(args)