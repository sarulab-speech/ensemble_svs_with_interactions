import argparse
import subprocess
import sys
from pathlib import Path

import fastdtw
import librosa
import matplotlib.pyplot as plt
import numpy as np
from util import HTSLabel


def label_alignment(
    source_timing_path: Path,
    source_wav_path: Path,
    target_wav_path: Path,
    out_target_label_path: Path,
):

    out_target_label_path.parent.mkdir(parents=True, exist_ok=True)

    # timings
    source_timing = HTSLabel()
    source_timing.load(source_timing_path)

    # wav
    source_wav, _ = librosa.load(str(source_wav_path), sr=16000, mono=True)
    target_wav, _ = librosa.load(str(target_wav_path), sr=16000, mono=True)

    # mfcc
    frame_shift_sec = 0.005
    kwargs = {
        "sr": 16000,
        "n_mfcc": 13,
        "hop_length": int(16000 * frame_shift_sec),
        "win_length": int(16000 * 0.025),
    }
    source_mfcc = librosa.feature.mfcc(y=source_wav, **kwargs).T
    target_mfcc = librosa.feature.mfcc(y=target_wav, **kwargs).T

    # dtw
    distance, path = fastdtw.fastdtw(source_mfcc, target_mfcc)
    path = np.asarray(path, dtype=int)

    # draw result
    plt.plot(path[:, 0], path[:, 1])
    plt.xlabel("source [frame]")
    plt.ylabel("target [frame]")
    plt.savefig(out_target_label_path.with_suffix(".pdf"))

    # convert time scale
    source_timing.change_time_scale("second")

    out_label = HTSLabel()
    out_label.time_scale = source_timing.time_scale

    for time, label in zip(source_timing.time, source_timing.label):
        time = list(map(lambda x: min(int(x / frame_shift_sec), path[-1, 0]), time))
        time = list(map(lambda x: np.mean(path[np.where(path[:, 0] == x), 1]), time))
        time = list(map(lambda x: x * frame_shift_sec, time))

        out_label.time.append(time)
        out_label.label.append(label)  # monophone

    # start and end
    out_label.time[0][0] = 0.0
    out_label.time[-1][1] = path[-1, 1] * frame_shift_sec

    out_label.save(out_target_label_path, delimiter="\t")


def parse_args():
    parser = argparse.ArgumentParser(
        description="generating label files from musicXML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "src_timing_path", type=str, help="filepath to timing file of source wav"
    )
    parser.add_argument("src_wav_path", type=str, help="filepath to source wav")
    parser.add_argument("tar_wav_path", type=str, help="filepath to target wav")
    parser.add_argument(
        "out_timing_path",
        type=str,
        help="filepath to timing file of target wav (output)",
    )
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    source_timing_path = Path(args.src_timing_path)
    source_wav_path = Path(args.src_wav_path)
    target_wav_path = Path(args.tar_wav_path)
    out_timing_path = Path(args.out_timing_path)

    label_alignment(
        source_timing_path, source_wav_path, target_wav_path, out_timing_path
    )

    print(f"Successfully generated {out_timing_path}.")
