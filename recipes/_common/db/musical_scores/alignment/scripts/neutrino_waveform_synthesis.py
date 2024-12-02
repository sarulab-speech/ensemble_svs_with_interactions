import argparse
import subprocess
import sys
from pathlib import Path

from util import (
    add_escape_mark,
    check_neutrino_directory,
    make_neutrino_preamble_command,
)


def normalize_full_labels(labels):
    return [x.replace("SYL", "").replace("ENG", "") for x in labels]


def neutrino_waveform_synthesis(
    in_label_full_path: Path,
    in_label_mono_path: Path,
    out_dir: Path,
    out_name: Path,
    neutrino_dir: Path,
    voice_name: str = "MERROW",
    n_thread: int = 3,
):

    out_path = out_dir.absolute()
    out_time_path = out_path / out_name.with_suffix(".lab")
    out_log_path = out_path / out_name.with_suffix(".log")
    out_lab_path = out_path / out_name.with_suffix(".timing")
    out_lf0_path = out_path / out_name.with_suffix(".f0")
    out_mgc_path = out_path / out_name.with_suffix(".mgc")
    out_bap_path = out_path / out_name.with_suffix(".bap")
    out_wav_path = out_path / out_name.with_suffix(".wav")

    for p in [
        out_time_path,
        out_lab_path,
        out_lf0_path,
        out_mgc_path,
        out_bap_path,
        out_wav_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # normalize labels
    labels = normalize_full_labels(open(in_label_full_path, "r").readlines())
    with open(out_lab_path, "w") as f:
        f.writelines(labels)

    # parameter generation
    NEUTRINO_PREAMBLE = make_neutrino_preamble_command(neutrino_dir)
    cmd = [
        neutrino_dir / "bin" / "NEUTRINO",
        out_lab_path,
        out_time_path,
        out_lf0_path,
        out_mgc_path,
        out_bap_path,
        str(neutrino_dir / "model" / voice_name) + "/",
        f"-n {n_thread}",
        "-k 0",
        "-t",
        "-m",
    ]
    cmd = " ".join([str(x) for x in cmd])
    try:
        r = subprocess.check_output(
            add_escape_mark(NEUTRINO_PREAMBLE + ";" + cmd),
            shell=True,
            cwd=neutrino_dir,
            stderr=subprocess.STDOUT,
        )
        with open(out_log_path, "w") as f:
            f.write(r.decode("utf-8"))
    except Exception as e:
        out_error_log_path = out_path / out_name.with_suffix(".errlog")
        with open(out_error_log_path, "w") as f:
            f.write(e.output.decode("utf-8"))
        raise ValueError(f"Failed to generate features for {in_label_full_path}.")

    # waveform synthesis
    cmd = [
        neutrino_dir / "bin" / "WORLD",
        out_lf0_path,
        out_mgc_path,
        out_bap_path,
        "-f 1.0",
        "-m 1.0",
        "-p 0.0",
        "-c 0.0",
        "-b 0.0",
        f"-n {n_thread}",
        "-t",
        f"-o {out_wav_path}",
    ]
    cmd = " ".join([str(x) for x in cmd])

    try:
        r = subprocess.check_output(
            add_escape_mark(NEUTRINO_PREAMBLE + ";" + cmd),
            shell=True,
            cwd=neutrino_dir,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        print(e)
        raise ValueError(f"Failed to sythesize waveform for {in_label_full_path}.")

    return out_wav_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="synthesize waveform from label files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--neutrino",
        type=str,
        default="/home/shinnosuke/Desktop/neutrino/NEUTRINO",
        help="dirname of NEUTRINO",
    )
    parser.add_argument(
        "--output", type=str, default="data/wav", help="dirname to save waveform"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="settings/voice/japanese.table",
        help="filename of voice setting",
    )
    parser.add_argument(
        "--n_thread", type=int, default=3, help="#threads to drive NEUTRINO"
    )
    parser.add_argument("label_path", type=str, help="filename of full-context file")
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    # load args
    in_label_full_path = Path(args.label_path)
    out_name = in_label_full_path.relative_to(
        in_label_full_path.parents[1]
    ).with_suffix("")
    in_label_mono_path = (
        in_label_full_path.parents[2] / "mono" / out_name.with_suffix(".lab")
    )

    out_dir = Path(args.output)
    neutrino_dir = Path(args.neutrino).absolute()
    n_thread = max(1, args.n_thread)

    # choose voice
    voice = {
        x.split(" ")[0]: x.strip("\n").split(" ")[1]
        for x in open(args.voice, "r").readlines()
    }

    voice_name = voice["default"]
    for part_name in set(voice.keys()) - set(["default"]):
        if part_name in in_label_full_path.stem:
            voice_name = voice[part_name]
            break

    if not check_neutrino_directory(neutrino_dir, voice_name=voice_name):
        raise ValueError("Please specify correct dir to find NEUTRINO.")

    out_wav_path = neutrino_waveform_synthesis(
        in_label_full_path,
        in_label_mono_path,
        out_dir,
        out_name,
        neutrino_dir,
        voice_name,
        n_thread,
    )

    print(
        f"Successfully synthesized {out_wav_path.relative_to(Path.cwd())} (voice: {voice_name})."
    )
