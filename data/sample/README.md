A youtube video will be used as a sample for alignment.

Title: Mejores Poemas - Mario Benedetti (Parte 1)

https://www.youtube.com/watch?v=M-Fokw3Wlco


To download the audio of the youtube video, run:

```bash
python -u ../../src/scripts/download_youtube_audio.py --id M-Fokw3Wlco
```

Reference text is also obtained from the youtube page. We provide the transcription for the M-Fokw3Wlco video in in **txt/** folder. In order to transform a *.txt* file to a *.tsv* file, run the next line:

```bash
python -u ../../src/scripts/txt_to_tsv.py --src_path data/sample/txt --dst_path data/sample/tsv --database_name benedetti
```