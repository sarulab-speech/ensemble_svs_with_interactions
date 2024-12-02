from pathlib import Path

import pretty_midi


def check_neutrino_directory(
    neutrino_dir: Path, language: str = "japanese", voice_name: str = "MERROW"
) -> bool:

    file_paths = [
        Path("settings") / "dic" / f"{language}.utf_8.conf",
        Path("settings") / "dic" / f"{language}.utf_8.table",
        Path("bin/NEUTRINO"),
        Path("bin/WORLD"),
        Path("bin/musicXMLtoLabel"),
        # Path("NSF/bin"),
        Path("model") / voice_name,
    ]

    file_exists = True
    for file_path in file_paths:
        if not (neutrino_dir / file_path).exists():
            print(f"Cannot find {file_path} in {neutrino_dir}.")
            file_exists = False

    return file_exists


def make_neutrino_preamble_command(neutrino_dir):
    return f"export LD_LIBRARY_PATH={neutrino_dir / 'bin'}:{neutrino_dir / 'NSF' / 'bin'}:$LD_LIBRARY_PATH"


def add_escape_mark(cmd):
    return cmd.replace("(", "\(").replace(")", "\)")


class HTSLabel:
    def __init__(self):
        self.label = []
        self.time = []
        self.time_scale = None

    def load(self, label_path, delimiter=" "):
        for line in open(label_path, "r").readlines():
            st, et, label = line.strip("\n").split(delimiter)
            st, et = float(st), float(et)

            if self.time_scale is None:
                self.time_scale = "htstime" if st.is_integer() else "second"

            self.label.append(label)
            self.time.append(
                list(map(int if self.time_scale == "htstime" else float, [st, et]))
            )

    def __len__(self):
        return len(self.label)

    def _htstime2second(self, htstime):
        return htstime * 1e-7

    def _second2htstime(self, second):
        return int(second * 1e7)

    def change_time_scale(self, target_scale):
        if self.time_scale == target_scale:
            return

        if self.time_scale == "htstime":
            self.time_scale = "second"
            self.time = [
                [self._htstime2second(x[0]), self._htstime2second(x[1])]
                for x in self.time
            ]
        elif self.time_scale == "second":
            self.time_scale = "htstime"
            self.time = [
                [self._second2htstime(x[0]), self._second2htstime(x[1])]
                for x in self.time
            ]

    def save(self, label_path, delimiter=" "):

        assert len(self.label) == len(self.time)

        with open(label_path, "w") as f:
            for time, label in zip(self.time, self.label):
                f.write(delimiter.join(list(map(str, time)) + [label]) + "\n")

    def del_dumplicates(self):
        while True:
            for i in range(1, len(self.time)):
                flag = (
                    (self.time[i - 1][0] == self.time[i][0])
                    and (self.time[i - 1][1] == self.time[i][1])
                    and (self.label[i - 1] == self.label[i])
                )
                if flag:
                    self.time.pop(i)
                    self.label.pop(i)
                    break
            if not flag:
                break

    def remove_small_fragments(self, threshold):
        while True:
            for i in range(len(self.time)):
                dur = self.time[i][1] - self.time[i][0]
                flag = dur < threshold
                if flag:
                    self.time.pop(i)
                    self.label.pop(i)
                    break
            if not flag:
                break


class FullContextHTSLabel(HTSLabel):
    CONTEXT_TAGS = ["phone"] + [_ for _ in "ABCDEFGHIJ"]

    def load(self, label_path, delimiter=" "):
        for line in open(label_path, "r").readlines():
            st, et, label = line.strip("\n").split(delimiter)
            st, et = float(st), float(et)

            if self.time_scale is None:
                self.time_scale = "htstime" if st.is_integer() else "second"

            self.label.append(self.parse_context(label))
            self.time.append(
                list(map(int if self.time_scale == "htstime" else float, [st, et]))
            )

    def parse_context(self, label):
        context_tags = self.CONTEXT_TAGS[1:]
        splitted_label = dict()
        start_index = 0
        for i, tag in enumerate(context_tags):
            end_index = label.find(f"/{tag}:")
            splitted_label["phone" if i == 0 else context_tags[i - 1]] = label[
                start_index:end_index
            ]
            start_index = end_index
        splitted_label[context_tags[-1]] = label[end_index:]
        return splitted_label

    def label_to_text(self, label):
        return "".join([label[_] for _ in self.CONTEXT_TAGS])

    def save(self, label_path, delimiter=" "):
        assert len(self.label) == len(self.time)

        with open(label_path, "w") as f:
            for time, label in zip(self.time, self.label):

                f.write(
                    delimiter.join(list(map(str, time)) + [self.label_to_text(label)])
                    + "\n"
                )


def create_midi_from_lab(filename: Path, out_label: HTSLabel):
    midi_obj = pretty_midi.PrettyMIDI()
    chorus_program = pretty_midi.instrument_name_to_program("Choir Aahs")
    chorus = pretty_midi.Instrument(program=chorus_program)
    for time, label in zip(out_label.time, out_label.label):
        if label != "xx":
            midi_note_num = pretty_midi.note_name_to_number(label)
            note = pretty_midi.Note(
                velocity=100, pitch=midi_note_num, start=time[0], end=time[1]
            )
            chorus.notes.append(note)
    midi_obj.instruments.append(chorus)
    midi_obj.write(str(filename))
