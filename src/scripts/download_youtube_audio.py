import os
import argparse
import youtube_dl


def download_audio_with_python_tool(link, filename, start, end):

    data_path = os.path.dirname(os.path.abspath(filename))
    
    # Set download options
    ydl_opts = {
        'audio-format': 'wav',
        'download_archive': data_path + '/downloaded_songs.txt',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }],
        'postprocessor_args': [
            '-format', 's16le',
            '-ar', '16000',
            '-ac', '1',
            '-ss', start,
            '-to', end
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        print('Downloading video: ' + link)
        ydl.cache.remove()
        ydl.download([link])


def download_audio_command_line(link, filename, start, end):

    tmp_filename = filename.replace('.wav', '_tmp.wav')

    # Download full video
    os.system("youtube-dl -o '{}' --extract-audio --audio-format wav {}".format(tmp_filename, link))

    # Get wanted audio
    os.system(
        "ffmpeg -i {} -loglevel warning -ac 1 -ar 16000 -format s16le -ss {} -to {} {}".format(
            tmp_filename, 
            start,
            end, 
            filename
            )
        )

    # Remove file
    os.system("rm " + tmp_filename)


def download_full_audio(link, filename):
    data_path = os.path.dirname(os.path.abspath(filename))
    
    # Set download options
    ydl_opts = {
        'audio-format': 'wav',
        'download_archive': data_path + '/downloaded_songs.txt',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }],
        'postprocessor_args': [
            '-format', 's16le',
            '-ar', '16000',
            '-ac', '1'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        print('Downloading video: ' + link)
        ydl.cache.remove()
        ydl.download([link])


def main(args):

    if args.id:

        # Compose the link
        link = 'https://www.youtube.com/watch?v=' + args.id
        filename = os.path.join(args.dst, 'Y_' + args.id)

        try: 
            #download_audio_command_line(link, filename, start, end)
            #download_audio_with_python_tool(link, filename, start, end)
            download_full_audio(link, filename)

        except KeyboardInterrupt:
            print('Interrupted')

    else:
        print("Empty video id.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scrip to download Audioset data")

    # Source Audioset data placed in the data folder of the project 
    parser.add_argument("--dst", help="destination directory", default="./audio_16kHz/")

    # Youtube video
    parser.add_argument("--id", help="Youtube video ID", default="") 

    args = parser.parse_args()

    main(args)