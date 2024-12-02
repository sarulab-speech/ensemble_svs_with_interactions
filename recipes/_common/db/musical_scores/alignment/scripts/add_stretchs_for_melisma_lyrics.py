import unicodedata
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from multiprocessing.sharedctypes import Value
from pathlib import Path

from tqdm import tqdm


def is_japanese(s):
    """日本語が含まれているかチェック

    Args:
        s (str): String

    Return:
        bool
    """
    for c in s:
        name = unicodedata.name(c)
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


class MusicXML:
    def __init__(self, filename):
        """

        Args:
            filename (str): MusicXML filename
        """
        with open(filename, "r", encoding="utf8") as fp:
            root = ET.fromstring(fp.read())
        self.root = root

    def add_pre_whole_rest(self):
        parts = self.root.findall("part")
        for p in self.root.find("part-list").findall("score-part"):
            part_name = p.find("part-name").text.replace(" ", "_")
            part_id = p.attrib["id"]
            part = list(filter(lambda x: x.attrib["id"] == part_id, parts))[0]
            if "perc" in part_name.lower() or "drum" in part_name.lower():
                continue
            part.insert(
                0,
                ET.fromstring(
                    f"""    <measure number="0" width="300">
    <attributes>
        <divisions>32</divisions>
    </attributes>
    <note>
        <rest measure="yes" />
        <duration>128</duration>
        <voice>1</voice>
        <type>whole</type>
        </note>
    </measure>"""
                ),
            )

    def add_stretchs(self):
        parts = self.root.findall("part")
        for p in self.root.find("part-list").findall("score-part"):
            part_name = p.find("part-name").text.replace(" ", "_")
            part_id = p.attrib["id"]
            part = list(filter(lambda x: x.attrib["id"] == part_id, parts))[0]
            if (
                "perc" in part_name.lower()
                or "drum" in part_name.lower()
                or "finger" in part_name.lower()
            ):
                continue
            prev_lyric = None
            # for note in part.findall(".//note"):
            #     # 休符は無視
            #     if note.find('rest') is not None:
            #         prev_lyric = None
            #         continue
            #     #
            #     lyric = note.find("lyric")
            #     if lyric is None:
            #         if prev_lyric is None:
            #             raised_measure_index = -1
            #             raised_note_index = -1
            #             for measure_index, measure in enumerate(part.findall(".//measure")):
            #                 for _note_index, _note in enumerate(measure.findall(".//note")):
            #                     if _note == note:
            #                         raised_measure_index = measure.attrib["number"]
            #                         raised_note_index = _note_index
            #                         break
            #             raise ValueError(f'Missing lyrics: note {raised_note_index} in measure {raised_measure_index}')
            #         new_lyric = ET.fromstring(ET.tostring(prev_lyric))
            #         new_lyric.remove(new_lyric.find("extend"))
            #         new_lyric.find("text").text = "ー"
            #         note.append(new_lyric)
            #     else:
            #         prev_lyric = note.find("lyric")

            for measure in part.findall(".//measure"):
                for note_index, note in enumerate(measure.findall(".//note")):
                    # 休符は無視
                    if note.find("rest") is not None:
                        prev_lyric = None
                        continue
                    # 装飾音はprev_lyricを保持したまま無視
                    if note.find("grace") is not None:
                        continue
                    #
                    lyric = note.find("lyric")
                    if lyric is None:
                        if prev_lyric is None:
                            raise ValueError(
                                f'Missing lyrics: note {note_index} in measure {measure.attrib["number"]}'
                            )
                        new_lyric = ET.fromstring(ET.tostring(prev_lyric))
                        try:
                            new_lyric.remove(new_lyric.find("extend"))
                        except:
                            print(ET.tostring(prev_lyric))
                            raise ValueError(
                                f'Missing lyrics: note {note_index} in measure {measure.attrib["number"]}'
                            )
                        new_lyric.find("text").text = "ー"
                        note.append(new_lyric)
                    else:
                        prev_lyric = note.find("lyric")

    def write(self, out_filename):
        """書き出し

        Args:
            out_filename (str): Output filename
        """
        out_filename.parent.mkdir(exist_ok=True, parents=True)
        tree = ET.ElementTree(self.root)
        tree.write(out_filename, encoding="utf8")


def main():
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs="*", help="Input musicxml files")
    args = parser.parse_args()

    for input_filename in [Path(_) for _ in args.inputs]:
        target = MusicXML(input_filename)
        target.add_pre_whole_rest()
        target.add_stretchs()
        target.write(input_filename)
        print(f"Successfully modified {input_filename}")


if __name__ == "__main__":
    main()
