import argparse
import subprocess
import sys
from pathlib import Path

from util import (
    add_escape_mark,
    check_neutrino_directory,
    make_neutrino_preamble_command,
)


def neutrino_label_synthesis(
    in_xml_path: Path, out_name: Path, out_dir: Path, neutrino_dir: Path
):
    XML2LABEL = neutrino_dir / "bin" / "musicXMLtoLabel"

    out_mono_path = (out_dir / "mono" / f"{out_name}.lab").absolute()
    out_full_path = (out_dir / "full" / f"{out_name}.lab").absolute()
    out_log_path = (out_dir / "log" / f"{out_name}.log").absolute()

    out_mono_path.parent.mkdir(parents=True, exist_ok=True)
    out_full_path.parent.mkdir(parents=True, exist_ok=True)
    out_log_path.parent.mkdir(parents=True, exist_ok=True)

    # execute Neutrino
    NEUTRINO_PREAMBLE = make_neutrino_preamble_command(neutrino_dir)
    cmd = f"{XML2LABEL} {in_xml_path.absolute()} {out_full_path} {out_mono_path}"
    print(add_escape_mark(NEUTRINO_PREAMBLE + ";" + cmd))
    print(neutrino_dir)
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
        raise ValueError(f"Failed to generate label files for {in_xml_path}.")

    return out_full_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="generating label files from musicXML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--neutrino", type=str, default="./NEUTRINO", help="dirname of NEUTRINO"
    )
    parser.add_argument(
        "--output", type=str, default="labels", help="dirname to save labels"
    )
    parser.add_argument("xml_path", type=str, help="filename of musicXML file")
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = parse_args()

    in_xml_path = Path(args.xml_path)
    out_name = in_xml_path.relative_to(in_xml_path.parents[1]).with_suffix("")
    out_dir = Path(args.output)
    neutrino_dir = Path(args.neutrino).absolute()

    if not check_neutrino_directory(neutrino_dir):
        raise ValueError("Please specify correct dir to find NEUTRINO.")

    out_label_path = neutrino_label_synthesis(
        in_xml_path, out_name, out_dir, neutrino_dir
    )

    print(f"Successfully generated {out_label_path.relative_to(Path.cwd())}.")
