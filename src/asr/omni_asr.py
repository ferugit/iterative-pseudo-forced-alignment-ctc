"""
Wrapper around facebook/omniASR-CTC-7B (omnilingual-asr / fairseq2) that
exposes the same interface as SpeechBrain's CTCSegmentation so the rest of
the alignment pipeline requires minimal changes.

Interface contract (matching the old speechbrain usage):

    aligner = OmniCTCAligner(model_card="omniASR_CTC_7B_v2", lang="spa_Latn")

    # normalise audio (replaces asr_model.audio_normalizer)
    audio_norm = aligner.audio_normalizer(audio_tensor, sample_rate)

    # get CTC log posteriors [T_frames, V]
    lpz = aligner.get_lpz(audio_norm)

    # prepare + run CTC segmentation
    task = aligner.prepare_segmentation_task(transcript, lpz, name, audio_len)
    segments = aligner.get_segments(task)
    task.set(**segments)
    lines = task.__str__().strip().split("\\n")

    # audio reduction factor (samples per CTC frame)
    ratio = aligner.estimate_samples_to_frames_ratio()

Notes
-----
* The forward-pass call `self.model(batch, None)` uses the fairseq2
  SequenceBatch API.  If the Wav2Vec2AsrModel signature differs in your
  installed version of omnilingual-asr, adjust _forward_pass() accordingly.

* OmniASR uses a character-level tokenizer across its 1 600+ languages.
  The blank token index is queried from the tokenizer vocab info at init time.
  If _build_char_list() raises, inspect tokenizer.vocab_info manually and
  override BLANK_TOKEN_ID.

* The model downloads to models/omni_asr/ (set via FAIRSEQ2_CACHE_DIR /
  HF_HOME).  Pass a different model_dir to override.

* The 40-second validation in ASRInferencePipeline.transcribe() is bypassed
  because we call the model directly.  Long windows (> 40 s) are processed
  without truncation; VRAM usage scales with audio length.
"""

import os
import torch
import torchaudio
import numpy as np

# Wav2vec2-style CNN encoder downsamples by 320 samples per frame at 16 kHz
# → ~50 frames per second.  Used to convert frame indices to timestamps.
_SAMPLES_PER_FRAME = 320
_SAMPLE_RATE = 16_000


class OmniCTCAligner:
    """
    Drop-in replacement for the (asr_model, CTCSegmentation) pair used in the
    alignment scripts.  A single instance handles both roles.
    """

    def __init__(
        self,
        model_card: str = "omniASR_CTC_7B_v2",
        lang: str = "spa_Latn",
        model_dir: str = "models/omni_asr",
        device: str | None = None,
    ):
        self.lang = lang
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Redirect model cache to project-local models/ directory
        os.makedirs(model_dir, exist_ok=True)
        os.environ.setdefault("FAIRSEQ2_CACHE_DIR", os.path.abspath(model_dir))
        os.environ.setdefault("HF_HOME", os.path.abspath(model_dir))

        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self._pipeline = ASRInferencePipeline(
            model_card=model_card,
            device=torch.device(self.device),
        )
        self._model = self._pipeline.model
        self._tokenizer = self._pipeline.tokenizer
        self._dtype = self._pipeline.dtype

        self.char_list = self._build_char_list()
        self.blank_id = self._find_blank_id()

    # ------------------------------------------------------------------
    # Public interface (mirrors speechbrain's CTCSegmentation + EncoderASR)
    # ------------------------------------------------------------------

    def audio_normalizer(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Resample to 16 kHz, convert to mono, amplitude-normalise.

        Parameters
        ----------
        audio : torch.Tensor [T_samples] or [T_samples, channels]
        sr    : int  original sample rate

        Returns
        -------
        torch.Tensor [T_samples_16k]  (1-D, CPU)
        """
        # Convert to mono first so resample always gets a 1-D tensor.
        # torchaudio.load with channels_first=False gives [T, C]; the default
        # channels_first=True gives [C, T].  We handle both by collapsing
        # whichever dimension is the smaller (channels are always <= 8).
        if audio.dim() == 2:
            if audio.shape[0] <= audio.shape[1]:
                audio = audio.mean(dim=0)   # [C, T] → [T]
            else:
                audio = audio.mean(dim=-1)  # [T, C] → [T]
        if sr != _SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, _SAMPLE_RATE)
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio.cpu()

    def get_lpz(self, audio_normalized: torch.Tensor) -> torch.Tensor:
        """
        Run the CTC encoder and return log posteriors.

        Parameters
        ----------
        audio_normalized : torch.Tensor [T_samples]
            16 kHz mono, amplitude-normalised.

        Returns
        -------
        torch.Tensor [T_frames, V]  on CPU
        """
        logits = self._forward_pass(audio_normalized)  # [T_frames, V]
        return torch.nn.functional.log_softmax(logits, dim=-1).cpu()

    def estimate_samples_to_frames_ratio(self) -> int:
        return _SAMPLES_PER_FRAME

    def prepare_segmentation_task(
        self,
        transcript: list[str],
        lpz: torch.Tensor,
        name: str,
        audio_len: int,
    ):
        """
        Build a SegmentationTask ready to be run by get_segments().

        Parameters
        ----------
        transcript : list[str]  list of utterance strings
        lpz        : torch.Tensor [T_frames, V]  log posteriors
        name       : str  utterance / file identifier
        audio_len  : int  audio length in samples (unused but kept for compat)
        """
        from ctc_segmentation import CtcSegmentationParameters, prepare_text, SegmentationTask

        config = CtcSegmentationParameters()
        config.char_list = self.char_list
        config.blank = self.blank_id
        config.index_duration = _SAMPLES_PER_FRAME / _SAMPLE_RATE

        ground_truth_mat, utt_begin_indices = prepare_text(config, transcript)

        task = SegmentationTask()
        task.name = name
        task.text = transcript
        task.config = config
        task.lpz = lpz.float().numpy()
        task.ground_truth_mat = ground_truth_mat
        task.utt_begin_indices = utt_begin_indices
        return task

    def get_segments(self, task) -> dict:
        """
        Run CTC segmentation and return timing / score data.

        Returns dict passed directly to task.set(**result).
        """
        from ctc_segmentation import ctc_segmentation

        timings, char_probs, state_list = ctc_segmentation(
            task.config, task.lpz, task.ground_truth_mat
        )
        return {
            "timings": timings,
            "char_probs": char_probs,
            "state_list": state_list,
            "utt_begin_indices": task.utt_begin_indices,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_pass(self, audio_normalized: torch.Tensor) -> torch.Tensor:
        """
        Run Wav2Vec2AsrModel forward pass and return raw logits [T_frames, V].

        fairseq2 0.6 API: model(source_seqs, batch_layout) → (logits, bl_out)
        BatchLayout encodes sequence lengths without a padding mask tensor.
        """
        from fairseq2.nn.batch_layout import BatchLayout

        audio = audio_normalized.unsqueeze(0).to(self.device, self._dtype)  # [1, T]
        seq_lens = torch.tensor([audio.shape[1]], device=self.device)
        batch_layout = BatchLayout(audio.shape, seq_lens=seq_lens, device=self.device)

        with torch.no_grad():
            logits, _ = self._model(audio, batch_layout)  # [1, T_frames, V]

        return logits.squeeze(0)  # [T_frames, V]

    def _build_char_list(self) -> list[str]:
        """
        Build the character vocabulary list from the omnilingual tokenizer.

        ctc_segmentation expects char_list[i] = string for token index i.
        """
        vocab_info = self._tokenizer.vocab_info
        vocab_size = vocab_info.size
        char_list = []
        for i in range(vocab_size):
            try:
                tok = self._tokenizer.model.index_to_token(i)
                char_list.append(tok if tok is not None else "")
            except Exception:
                char_list.append("")
        return char_list

    def _find_blank_id(self) -> int:
        """
        Return the index of the CTC blank token.

        In fairseq2 CTC models the blank token is usually the pad token
        (index 0).  We also check common token strings ("<blank>", "<pad>").
        """
        # Check if vocab_info exposes a pad_idx attribute
        vocab_info = self._tokenizer.vocab_info
        if hasattr(vocab_info, "pad_idx") and vocab_info.pad_idx is not None:
            return vocab_info.pad_idx

        # Fall back to searching by string
        for blank_str in ("<blank>", "<pad>", "[blank]"):
            try:
                idx = self._tokenizer.model.token_to_index(blank_str)
                if idx is not None:
                    return idx
            except Exception:
                pass

        return 0  # wav2vec2 CTC models conventionally use index 0
