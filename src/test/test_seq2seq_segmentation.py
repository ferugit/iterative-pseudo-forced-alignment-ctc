from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.alignment.ctc_segmentation import CTCSegmentation

def main():

    # ASR parameters 
    source_path = 'config/'
    hparams_path = 'seq2seq_sp_with_wav2vec.yaml'
    savedir_path = 'data/asr/seq2seq/savedir'
    
    # Audio Path
    audio_file = 'data/audio/test/single_word.wav'
    
    # STT
    asr_model = EncoderDecoderASR.from_hparams(source=source_path, hparams_file=hparams_path, savedir=savedir_path) 
    transcription = asr_model.transcribe_file(audio_file)
    print(transcription)

    # CTC segmentation
    kwargs = {
        'min_window_size': 8000,
        'max_window_size': 100000,
        'gratis_blank': True
    }

    aligner = CTCSegmentation(asr_model, kaldi_style_text=False, time_stamps="fixed", **kwargs)
    print(aligner.estimate_samples_to_frames_ratio())
    text = ["AUXILIO"]
    segments = aligner(audio_file, text, name="CommonVoiceSP")
    print(segments)

if __name__ == "__main__":
    main()
