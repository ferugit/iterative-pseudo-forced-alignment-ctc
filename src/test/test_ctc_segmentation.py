from speechbrain.pretrained import EncoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation

def main():

    # ASR parameters 
    source_path = 'config/'
    hparams_path = 'ctc_sp_with_wav2vec.yaml'
    savedir_path = 'data/asr/ctc/savedir'
    
    # Audio Path
    audio_file = 'data/audio/test/single_word.wav'
    
    # STT
    asr_model = EncoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path) 
    transcription = asr_model.transcribe_file(audio_file)
    print(transcription)

    # CTC segmentation: configuration from speechbrain
    kwargs = {
        'min_window_size': 80000,
        'max_window_size': 100000,
        'gratis_blank': False,
        'set_blank': 0
    }

    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, **kwargs)

    # CTC segmentation: low-level configuration
    aligner.config.space = "·"
    aligner.config.replace_spaces_with_blanks = True
    aligner.config.blank_transition_cost_zero = True
    aligner.config.preamble_transition_cost_zero = True
    aligner.config.backtrack_from_max_t = True
    aligner.config.self_transition = "ε"
    aligner.config.start_of_ground_truth = "#"
    aligner.config.excluded_characters = ".,»«•❍·"
    aligner.config.tokenized_meta_symbol = "▁"

    print(aligner.estimate_samples_to_frames_ratio())
    text = ["AUXILIO"]
    segments = aligner(audio_file, text, name="CommonVoiceSP")
    print(segments)

if __name__ == "__main__":
    main()
