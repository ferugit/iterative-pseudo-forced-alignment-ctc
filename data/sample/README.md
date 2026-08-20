A YouTube video is used as sample data for alignment.

**Title:** Mejores Poemas - Mario Benedetti (Parte 1)
**URL:** https://www.youtube.com/watch?v=M-Fokw3Wlco

To download the audio, run from the repository root:

```bash
yt-dlp -x --audio-format wav \
    --postprocessor-args "-ar 16000 -ac 1" \
    -o "data/sample/audio_16kHz/Y_M-Fokw3Wlco.%(ext)s" \
    "https://www.youtube.com/watch?v=M-Fokw3Wlco"
```

`yt-dlp` is available in the virtual environment after `pip install -r requirements.txt`.
The original `download_youtube_audio.py` script used `youtube-dl`, which is no longer
maintained and fails on current YouTube. Use `yt-dlp` directly as shown above.

The reference transcription for this video is provided in `txt/Y_M-Fokw3Wlco.txt`
(YouTube closed-captions). The corresponding `tsv/benedetti.tsv` was generated from it
and is already included — you only need to download the audio.
