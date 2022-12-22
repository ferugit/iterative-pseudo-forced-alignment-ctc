import sys
sys.path.insert(1, sys.path[0].replace('/src/test', ''))

import torchaudio

from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

def main():

    # Load audio
    filename = 'data/audio/test/CA-20171025_30seg.wav'
    audio, sr = torchaudio.load(filename)
    audio = (audio[0] + audio[1])/2.0
    print(audio.shape)
    model_hub = "facebook/wav2vec2-large-xlsr-53-spanish"
    #model_hub = "data/asr/ctc/38_tokenizer/wav2vec2_checkpoint/wav2vec2.ckpt"
    save_path = "data/asr/ctc/38_tokenizer/wav2vec2_checkpoint"
    model = HuggingFaceWav2Vec2(model_hub, save_path)
    commonvoice_mean_length = 5.0
    commonvoice_mean_samples = int(commonvoice_mean_length*sr)
    steps = int(audio.shape[0]/commonvoice_mean_samples)
    for index in range(1, steps+1):
        audio_chunk = audio[:index*commonvoice_mean_samples]
        print(audio_chunk.shape)
        outputs = model(audio_chunk.unsqueeze(0))
        print(outputs.shape) # [batch, t_bins, x_bins]
        print(outputs[0,0,:10])

if __name__ == '__main__':
    main()