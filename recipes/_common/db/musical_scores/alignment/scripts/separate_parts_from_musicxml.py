from __future__ import division

import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

import numpy
from tqdm import tqdm


def get_part_id_and_names(filename):
    with open(filename, "r", encoding="utf8") as fp:
        root = ET.fromstring(fp.read())

    part_id_list = []
    for p in root.find("part-list").findall("score-part"):
        part_name = p.find("part-name").text.replace(" ", "_")
        part_id = p.attrib["id"]
        part_id_list.append((part_id, part_name))

    del root
    return part_id_list


def get_bpms(filename):
    with open(filename, "r", encoding="utf8") as fp:
        root = ET.fromstring(fp.read())
    for p in root.findall("part"):
        bpm_templates = []
        # check divisions
        divisions_list = p.findall("measure/attributes/divisions")
        assert (
            len(divisions_list) == 1
        ), f"Varying divisions in one part are not supported"
        divisions = int(divisions_list[0].text)
        #
        for child in p.iter("measure"):
            metronomes = child.findall("direction/direction-type/metronome")
            if len(metronomes) > 0:
                note_count = 0
                cur_div = 0
                for i in child:
                    if i.tag == "note":
                        j = i.find("rest")
                        if j is None:
                            note_count += 1
                        cur_div += int(i.find("duration").text)
                    if len(i.findall("direction-type/metronome")) == 1:
                        # assert note_count == 0, f'BPM after notes not supported at measure {child.attrib["number"]}'
                        # bpm_templates.append((child.attrib["number"], ET.tostring(i), note_count)) # measure number, element string, note_count
                        metronome_position_in_div = cur_div
                        offset_elem = i.find("direction-type/offset")
                        if (offset_elem is not None) and (
                            "sound" in offset_elem.attrib
                        ):
                            if offset_elem.attrib["sound"] == "yes":
                                metronome_position_in_div += int(offset_elem.text)
                        bpm_templates.append(
                            (
                                child.attrib["number"],
                                ET.tostring(i),
                                note_count,
                                metronome_position_in_div / float(divisions),
                            )
                        )  # measure number, element string, note_count, position
                    elif len(i.findall("direction-type/metronome")) > 1:
                        raise NotImplementedError(
                            f"Mutiple metronome elements in one measure are not supported"
                        )

        if len(bpm_templates) > 0:
            break
    return bpm_templates


def is_bpm_equal(b1, b2):
    if b1[0] != b2[0]:
        return False  # measure is different
    root1 = ET.fromstring(b1[1])
    root2 = ET.fromstring(b2[1])
    beat_unit1 = root1.find(".//beat-unit").text
    beat_unit2 = root2.find(".//beat-unit").text
    per_minute1 = root1.find(".//per-minute").text
    per_minute2 = root2.find(".//per-minute").text
    return (beat_unit1 == beat_unit2) and (
        per_minute1 == per_minute2
    )  # beat is different or same


def separate_score_parts(part_id_list, bpm_templates, filename, output_dir, args=None):
    for part_id, part_name in part_id_list:
        with open(filename, "r", encoding="utf-8") as fp:
            root = ET.fromstring(fp.read())
        # remove score-part
        p_list = root.find("part-list")
        for p in p_list.findall("score-part"):
            if part_id != p.attrib["id"]:
                p_list.remove(p)
        # remove part-group
        for p in p_list.findall("part-group"):
            p_list.remove(p)
        # remove parts
        for p in root.findall("part"):
            if p.attrib["id"] != part_id:
                root.remove(p)
        ## get divisions
        measures = root.find("part").findall("measure")
        min_measure_num = numpy.min(
            [int(measure.attrib["number"]) for measure in measures]
        )
        first_measure = list(
            filter(lambda x: int(x.attrib["number"]) == min_measure_num, measures)
        )[0]
        divisions = int(
            first_measure.find("attributes/divisions").text
        )  # num. of counts of a quarter note
        # insert bpm_templates
        if len(root.findall(".//metronome")) == 0:
            # no bpm directions
            for m in root.iter("measure"):
                measure_number = m.attrib["number"]
                matched_list = list(
                    filter(lambda x: x[0] == measure_number, bpm_templates)
                )  # list of measure number, element string, note_count, position
                if len(matched_list) == 1:
                    direction_including_metronome_element = ET.fromstring(
                        matched_list[0][1]
                    )
                    if matched_list[0][3] > 0:
                        # if direction is applied at the intermediate of the measure
                        offset_val_in_div = int(matched_list[0][3] * divisions)
                        if args.debug:
                            print(f"#### OFFSET: {offset_val_in_div}")
                            print(matched_list[0])
                        direction_including_metronome_element.find("offset").text = str(
                            offset_val_in_div
                        )
                        #
                        cur_div = 0
                        inserted_index = 0
                        for index, n in enumerate(m):
                            if n.tag == "note":
                                cur_div += int(n.find("duration").text)
                            if cur_div == offset_val_in_div:
                                inserted_index = index + 1
                                break
                            elif cur_div > offset_val_in_div:
                                # Set "sound" attrib (in offset tag) to "yes" so that it can affects the sound timing (not only visualization in scores).
                                inserted_index = 0
                                direction_including_metronome_element.find(
                                    "offset"
                                ).attrib["sound"] = "yes"
                                break
                                # raise NotImplementedError(f'Tempo change in a performed note does not support: current position in div {cur_div}, offset {offset_val_in_div}')
                        if args.debug:
                            print(
                                f"### inserted at {inserted_index}: {cur_div} vs {offset_val_in_div}"
                            )
                        # insert direction element including metronome element
                        m.insert(inserted_index, direction_including_metronome_element)
                    else:
                        # insert direction element including metronome element as the first element
                        m.insert(0, direction_including_metronome_element)
                elif len(matched_list) == 0:
                    pass
                else:
                    for _ in matched_list:
                        print(str(_))
                    raise NotImplementedError
        # insert implicitly indicated rest in first measure
        # assert divisions%4==0 or divisions==2
        beats = int(first_measure.find("attributes/time/beats").text)
        beat_type = int(first_measure.find("attributes/time/beat-type").text)
        annotated_note_durations = numpy.sum(
            [int(_.text) for _ in first_measure.findall("note/duration")]
        )
        #####
        if beat_type == 4:  # 1/4, 2/4, 3/4, 4/4, ...
            divisions_for_quarter = divisions
            measure_duration = divisions_for_quarter * beats
        elif beats == 2 and beat_type == 2:  # 2/2
            divisions_for_half = divisions * 2
            measure_duration = divisions_for_half * beats
        elif beats == 6 and beat_type == 8:  # 6/8
            # divisions: num. of counts of a quarter note
            divisions_for_eighth = divisions // 2
            measure_duration = divisions_for_eighth * beats
        elif beats == 9 and beat_type == 8:  # 9/8
            # divisions: num. of counts of a quarter note
            divisions_for_eighth = divisions // 2
            measure_duration = divisions_for_eighth * beats
        else:
            raise NotImplementedError
        #####
        rest_duration = measure_duration - annotated_note_durations
        if rest_duration > 0:
            for i, e in enumerate(first_measure):
                if e.tag == "note":
                    first_measure.insert(
                        i,
                        ET.fromstring(
                            f"""<note>
        <rest />
        <duration>{rest_duration:d}</duration>
        </note>

"""
                        ),
                    )
                    break
        # dump
        out_filename = (
            output_dir
            / f"{filename.stem}_parts"
            / f"{filename.stem}-{part_name}.musicxml"
        )
        out_filename.parent.mkdir(exist_ok=True, parents=True)
        tree = ET.ElementTree(root)
        tree.write(out_filename, encoding="utf8", xml_declaration=True)
        del root
        del tree
        print(f"Successfully generate {out_filename}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True, help="Input musicxml files"
    )
    parser.add_argument("-o", "--outdir", required=True, help="Output dir")
    parser.add_argument("--debug", action="store_true", help="Debug mode flag")
    args = parser.parse_args()

    output_dir = Path(args.outdir)

    for input_musicxml_filename in [Path(_) for _ in args.inputs]:
        part_id_list = get_part_id_and_names(input_musicxml_filename)
        bpm_templates = get_bpms(input_musicxml_filename)
        separate_score_parts(
            part_id_list, bpm_templates, input_musicxml_filename, output_dir, args=args
        )


if __name__ == "__main__":
    main()
