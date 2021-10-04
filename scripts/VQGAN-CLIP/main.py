import glob
import subprocess
from subprocess import PIPE
from subprocess import Popen

import librosa
from PIL import Image
from tqdm import tqdm

# import os
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


def generate_image_slides(audio_file, out_file_prefix, frame_size=None, hop_size=None):
    audio, sr = librosa.load(audio_file, sr=16000)
    if frame_size and hop_size:
        num_frames = int(audio.shape[0] / hop_size) - 1
        assert num_frames > 1
        for i in range(num_frames):
            if i == 0:
                subprocess.call(
                    [
                        "python",
                        "generate.py",
                        "-p",
                        "",
                        "-ap",
                        audio_file,
                        "-aframe",
                        "{}".format(frame_size),
                        "-ahop",
                        "{}".format(hop_size),
                        "-ai",
                        "{}".format(i),
                        "-o",
                        "{}_{}.png".format(out_file_prefix, i),
                    ]
                )
            elif i > 0:
                subprocess.call(
                    [
                        "python",
                        "generate.py",
                        "-p",
                        "",
                        "-ap",
                        audio_file,
                        "-ahop",
                        "{}".format(hop_size),
                        "-ai",
                        "{}".format(i),
                        "-ii",
                        "{}_{}.png".format(out_file_prefix, i - 1),
                        "-o",
                        "{}_{}.png".format(out_file_prefix, i),
                    ]
                )
            else:
                assert False
    else:
        subprocess.call(
            [
                "python",
                "generate.py",
                "-p",
                "",
                "-ap",
                audio_file,
                "-o",
                "{}.png".format(out_file_prefix),
            ]
        )
    return None


def generate_interpolate_video(image_file_prefix, video_only_file, fps=10):
    img_files = sorted(
        glob.glob("{}_*.png".format(image_file_prefix)),
        key=lambda x: int(x.split(".")[-2].split("_")[1]),
    )

    frames = []
    for f in img_files:
        temp = Image.open(f)
        keep = temp.copy()
        frames.append(keep)
        temp.close()

    ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps=30'"
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            "{}".format(fps),
            "-i",
            "-",
            "-b:v",
            "10M",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-strict",
            "-2",
            "-filter:v",
            f"{ffmpeg_filter}",
            video_only_file,
        ],
        stdin=PIPE,
    )

    for im in tqdm(frames):
        im.save(p.stdin, "PNG")
    p.stdin.close()
    p.wait()


def mix_audio_video(audio_file, video_only_file, audio_video_file):
    cmd = 'ffmpeg -i {} -i "{}" -c:v copy -c:a aac {}'.format(
        video_only_file, audio_file, audio_video_file
    )
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    audio_file = "music.wav"
    image_file_prefix = audio_file[:-4]
    video_only_file = "{}_no_audio.mp4".format(image_file_prefix)
    audio_video_file = "{}.mp4".format(image_file_prefix)

    # generate image
    generate_image_slides(audio_file, image_file_prefix)

    # generate video
    generate_image_slides(
        audio_file, image_file_prefix, frame_size=16000, hop_size=1600
    )
    generate_interpolate_video(image_file_prefix, video_only_file, fps=10)
    mix_audio_video(audio_file, video_only_file, audio_video_file)
