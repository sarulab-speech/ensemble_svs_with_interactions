import argparse
import os
import re
import sys
from glob import glob
from os.path import basename, exists, join, splitext

import librosa
import numpy as np
import soundfile as sf
import yaml
from nnmnkwii.io import hts
from nnsvs.io.hts import get_note_indices
from tqdm import tqdm


def compute_nosil_duration(lab, threshold=5.0):
    is_full_context = "@" in lab[0][-1]
    sum_d = 0
    for s, e, label in lab:
        d = (e - s) * 1e-7
        if is_full_context:
            is_silence = "-sil" in label or "-pau" in label
        else:
            is_silence = label == "sil" or label == "pau"
        if is_silence and d > threshold:
            pass
        else:
            sum_d += d
    return sum_d


def fix_offset(lab):
    offset = lab.start_times[0]
    lab.start_times = np.asarray(lab.start_times) - offset
    lab.end_times = np.asarray(lab.end_times) - offset
    return lab


def _is_silence(label):
    is_full_context = "@" in label
    if is_full_context:
        is_silence = "-sil" in label or "-pau" in label
    else:
        is_silence = label == "sil" or label == "pau"
    return is_silence


def remove_sil_and_pau(lab):
    newlab = hts.HTSLabelFile()
    for label in lab:
        if "-sil" not in label[-1] and "-pau" not in label[-1]:
            newlab.append(label, strict=False)

    return newlab


def sanity_check_lab(lab):
    for b, e, _ in lab:
        assert e - b > 0


def trim_sil_and_pau(lab, return_indices=False):
    forward = 0
    while "-sil" in lab.contexts[forward] or "-pau" in lab.contexts[forward]:
        forward += 1

    backward = len(lab) - 1
    while "-sil" in lab.contexts[backward] or "-pau" in lab.contexts[backward]:
        backward -= 1

    if return_indices:
        return lab[forward : backward + 1], forward, backward
    else:
        return lab[forward : backward + 1]


def trim_long_sil_and_pau(lab, return_indices=False, threshold=10.0):
    forward = 0
    while True:
        d = (lab.end_times[forward] - lab.start_times[forward]) * 1e-7
        if _is_silence(lab.contexts[forward]) and d > threshold:
            forward += 1
        else:
            break

    backward = len(lab) - 1
    while True:
        d = (lab.end_times[backward] - lab.start_times[backward]) * 1e-7
        if _is_silence(lab.contexts[backward]) and d > threshold:
            backward -= 1
        else:
            break

    if return_indices:
        return lab[forward : backward + 1], forward, backward
    else:
        return lab[forward : backward + 1]


def segment_labels(
    lab, strict=True, threshold=1.0, min_duration=5.0, force_split_threshold=10.0
):
    """Segment labels based on sil/pau

    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]

    """
    segments = []
    seg = hts.HTSLabelFile()
    start_indices = []
    end_indices = []
    si = 0
    large_silence_detected = False

    for idx, (s, e, label) in enumerate(lab):
        d = (e - s) * 1e-7
        is_silence = _is_silence(label)

        if len(seg) > 0:
            # Compute duration except for long silences
            seg_d = compute_nosil_duration(seg)
        else:
            seg_d = 0

        # let's try to split
        # if we find large silence, force split regardless min_duration
        if (d > force_split_threshold) or (
            is_silence and d > threshold and seg_d > min_duration
        ):
            if idx == len(lab) - 1:
                continue
            elif len(seg) > 0:
                if d > force_split_threshold:
                    large_silence_detected = True
                else:
                    large_silence_detected = False
                start_indices.append(si)
                si = 0
                end_indices.append(idx - 1)
                segments.append(seg)
                seg = hts.HTSLabelFile()
            continue
        else:
            if len(seg) == 0:
                si = idx
            seg.append((s, e, label), strict)

    if len(seg) > 0:
        seg_d = compute_nosil_duration(seg)
        # If the last segment is short, combine with the previous segment.
        if seg_d < min_duration and not large_silence_detected:
            end_indices[-1] = si + len(seg) - 1
        else:
            start_indices.append(si)
            end_indices.append(si + len(seg) - 1)

    #  Trim large sil for each segment
    segments2 = []
    start_indices_new, end_indices_new = [], []
    for s, e in zip(start_indices, end_indices):
        seg = lab[s : e + 1]

        # ignore "sil" or "pau" only segment
        only_pau = True
        for idx in range(len(seg)):
            if not _is_silence(seg.contexts[idx]):
                only_pau = False
                break
        if only_pau:
            continue
        seg2, forward, backward = trim_long_sil_and_pau(seg, return_indices=True)

        start_indices_new.append(s + forward)
        end_indices_new.append(s + backward)

        segments2.append(seg2)

    return segments2, start_indices_new, end_indices_new


def segment_multitrack_labels(labels):
    """
    return positions at which labels are segmented
    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]

    """
    pattern = re.compile(r"-(.+?)\+")
    nonsil_timings = []
    for label in labels:
        for (s, e, full) in label:
            match = pattern.search(full)
            assert match
            p = match.group(1)
            if p != "pau" and p != "sil":
                continue
            nonsil_timings.append((s, -1))
            nonsil_timings.append((e, 1))

    nonsil_timings = sorted(nonsil_timings)

    nonsil_count = len(labels)  # the number of singers singing at a current timing t
    cut_positions = [
        max([l[0][0] for l in labels])
    ]  # note that the first time may be >0
    for idx, (t, x) in enumerate(nonsil_timings):
        nonsil_count += x

        # skip if next element has the same timing
        if idx < len(nonsil_timings) - 1:
            nxt, _ = nonsil_timings[idx + 1]
            if t == nxt:
                continue

        if nonsil_count == 0 and t > 0 and t - cut_positions[-1] >= 10000000:  #
            cut_positions.append(t)
        elif t - cut_positions[-1] >= 8 * (10 ** 7):  # 8秒を超える場合はカット
            cut_positions.append(t)

    # if the lengths of each part are different, use the shortest one
    cut_positions[-1] = min([l[-1][1] for l in labels])

    for i, p in enumerate(cut_positions):
        if i < len(cut_positions) - 1:
            assert (
                p < cut_positions[i + 1]
            ), f"{p}, {cut_positions[i + 1]}, {cut_positions}, {nonsil_timings}, {len(cut_positions)}"
    return cut_positions


def segment_label_from_position(label, cut_positions, strict):
    """
    return
    segments: segments split by cut_positions
    indices: start_idx and end_idx of each segments

    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]
    [(0,2), (3,5), (6,8), (9,11)]

    """
    segments = []
    indices = []
    for i in range(len(cut_positions) - 1):
        l_p = cut_positions[i]
        r_p = cut_positions[i + 1]

        seg = hts.HTSLabelFile()
        l_idx = 1e9
        r_idx = -1e9
        for idx, (s, e, p) in enumerate(label):  # [s, e)
            if s < r_p and l_p < e:
                seg.append((max(s, l_p), min(e, r_p), p), strict)
                l_idx = min(l_idx, idx)
                r_idx = max(r_idx, idx)
        segments.append(seg)
        indices.append((l_idx, r_idx))
    return segments, indices


def segment_label_from_indices(label, indices):
    """
    return
    segments: segments split by indices

    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]

    """
    segments = []

    for (s, e) in indices:
        segments.append(label[s : e + 1])
    return segments


def get_parser():
    parser = argparse.ArgumentParser(
        description="Data preparation for jaCappella",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("jaCappella_root", type=str, help="jaCappella dir")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("config_path", type=str, help="config.yaml")
    return parser


args = get_parser().parse_args(sys.argv[1:])

out_dir = args.out_dir
config_path = args.config_path
with open(config_path, "r") as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)
if config is None:
    print(f"Cannot read config file: {sys.argv[1]}.")
    sys.exit(-1)

ok_count = 0
ng_count = 0
speakers = config["spk_list"]
songs = []

for spk in speakers:
    # Make aligned full context labels
    full_align_dir = join(out_dir, "label_phone_align")
    full_score_dir = join(out_dir, "label_phone_score")
    for d in [full_align_dir, full_score_dir]:
        os.makedirs(d, exist_ok=True)

    # mono
    align_lab_files = sorted(glob(join(args.jaCappella_root, spk, "*_aligned.lab")))

    # full
    score_lab_files = sorted(glob(join(args.jaCappella_root, spk, "*_score.lab")))

    for align_path, score_path in zip(align_lab_files, score_lab_files):
        align_mono_lab = hts.load(align_path)
        name = basename(align_path)
        name = re.search(r"(.*)_", name).group(1)
        if name not in songs:
            songs.append(name)

        score_full_lab = hts.load(score_path)

        assert len(align_mono_lab) == len(score_full_lab), f"{align_path} {score_path}"

        # rounding
        has_too_short_ph = False
        for idx in range(len(align_mono_lab)):
            b, e = align_mono_lab.start_times[idx], align_mono_lab.end_times[idx]
            b *= 10000000
            e *= 10000000
            bb, ee = round(b / 50000) * 50000, round(e / 50000) * 50000
            if bb >= ee:
                # ensure minimum frame length 1
                align_mono_lab.end_times[idx] = align_mono_lab.start_times[idx] + 50000
                align_mono_lab.start_times[idx + 1] = align_mono_lab.end_times[idx]
                has_too_short_ph = True

        if has_too_short_ph is False or config["exclude_too_short_frame"] is False:
            # gen full-context
            align_full_lab = hts.HTSLabelFile()
            new_score_full_lab = hts.HTSLabelFile()
            for idx, label in enumerate(score_full_lab):
                b, e, c = label
                b, e = round(int(b) / 50000) * 50000, round(int(e) / 50000) * 50000
                assert b != e
                new_score_full_lab.append((b, e, c), strict=False)

                b, e = align_mono_lab.start_times[idx], align_mono_lab.end_times[idx]
                align_full_lab.append((b, e, c), strict=False)
            with open(join(full_score_dir, spk + "_" + name + ".lab"), "w") as of:
                of.write(str(new_score_full_lab))
            with open(join(full_align_dir, spk + "_" + name + ".lab"), "w") as of:
                of.write(str(align_full_lab))
            print(align_path, "IS OK.", b, e)
            ok_count += 1
        else:
            print(align_path, "has too short frame.", b, e)
            ng_count += 1
print("OK:", ok_count, ",NG:", ng_count)

songs = sorted(songs)
for song_name in songs:
    # divide .lab into small segments
    lengths = {}

    # divide *aligned.lab
    suffix = "_" + song_name + ".lab"
    a_files = sorted(glob(join(out_dir, "label_phone_align", f"*{suffix}")))
    s_files = sorted(glob(join(out_dir, "label_phone_score", f"*{suffix}")))

    a_labels = []
    for label_path in a_files:
        a_labels.append(hts.load(label_path))
    s_labels = []
    for label_path in s_files:
        s_labels.append(hts.load(label_path))

    cut_positions = segment_multitrack_labels(a_labels)

    for a_label_path, s_label_path in zip(a_files, s_files):
        utt_id = splitext(basename(a_label_path))[0]
        assert utt_id == splitext(basename(s_label_path))[0]
        a_label = hts.load(a_label_path)
        s_label = hts.load(s_label_path)

        print(a_label_path)
        base_segments, indices = segment_label_from_position(
            a_label, cut_positions, True
        )

        # keep lengths of each segment(align) for logging
        d = []
        for seg in base_segments:
            d.append((seg.end_times[-1] - seg.start_times[0]) * 1e-7)
        lengths[utt_id] = d

        dst_dir = join(out_dir, "label_phone_align_seg")
        os.makedirs(dst_dir, exist_ok=True)
        for idx, seg in enumerate(base_segments):
            with open(join(dst_dir, f"{utt_id}_seg{idx}.lab"), "w") as of:
                of.write(str(seg))

        base_segments = segment_label_from_indices(s_label, indices)
        dst_dir = join(out_dir, "label_phone_score_seg")
        os.makedirs(dst_dir, exist_ok=True)
        for idx, seg in enumerate(base_segments):
            with open(join(dst_dir, f"{utt_id}_seg{idx}.lab"), "w") as of:
                of.write(str(seg))

    if len(lengths) == 0:
        continue
    for ls in [lengths]:
        for k, v in ls.items():
            print(
                "{}.lab: segment duration min {:.02f}, max {:.02f}, mean {:.02f}".format(
                    k, np.min(v), np.max(v), np.mean(v)
                )
            )

        flatten_lengths = []
        for k, v in ls.items():
            sys.stdout.write(f"{k}.lab: segment lengths: ")
            for d in v:
                sys.stdout.write("{:.02f}, ".format(d))
                flatten_lengths.append(d)
            sys.stdout.write("\n")

        print(
            "Segmentation stats: min {:.02f}, max {:.02f}, mean {:.02f}".format(
                np.min(flatten_lengths),
                np.max(flatten_lengths),
                np.mean(flatten_lengths),
            )
        )

        print("Total number of segments: {}".format(len(flatten_lengths)))

    # Prepare data for time-lag models
    dst_dir = join(out_dir, "timelag")
    lab_align_dst_dir = join(dst_dir, "label_phone_align")
    lab_score_dst_dir = join(dst_dir, "label_phone_score")

    for d in [lab_align_dst_dir, lab_score_dst_dir]:
        os.makedirs(d, exist_ok=True)

    print("Prepare data for time-lag models")
    full_lab_align_files = sorted(glob(join(full_align_dir, f"*{suffix}")))
    full_lab_score_files = sorted(glob(join(full_score_dir, f"*{suffix}")))
    black_list = []
    for lab_align_path, lab_score_path in zip(
        full_lab_align_files, full_lab_score_files
    ):
        name = basename(lab_align_path)
        utt_id = splitext(basename(lab_align_path))[0]

        lab_align = hts.load(lab_align_path)
        lab_score = hts.load(lab_score_path)

        # this may harm for computing offset
        lab_align = remove_sil_and_pau(lab_align)
        lab_score = remove_sil_and_pau(lab_score)

        # Extract note onsets and let's compute a offset
        note_indices = get_note_indices(lab_score)

        onset_align = np.asarray(lab_align[note_indices].start_times)
        onset_score = np.asarray(lab_score[note_indices].start_times)

        global_offset = (onset_align - onset_score).mean()
        global_offset = int(round(global_offset / 50000) * 50000)

        # Apply offset correction only when there is a big gap
        apply_offset_correction = (
            np.abs(global_offset * 1e-7) > config["offset_correction_threshold"]
        )
        if apply_offset_correction:
            print(f"{name}: Global offset (in sec): {global_offset * 1e-7}")
            lab_score.start_times = list(
                np.asarray(lab_score.start_times) + global_offset
            )
            lab_score.end_times = list(np.asarray(lab_score.end_times) + global_offset)
            onset_score += global_offset

        seg_idx = 0
        while True:
            lab_align_path = join(full_align_dir + "_seg", f"{utt_id}_seg{seg_idx}.lab")
            lab_score_path = join(full_score_dir + "_seg", f"{utt_id}_seg{seg_idx}.lab")
            name = basename(lab_align_path)
            assert seg_idx > 0 or exists(lab_align_path)
            if not exists(lab_align_path):
                break
            assert exists(lab_score_path)

            lab_align = hts.load(lab_align_path)
            lab_score = hts.load(lab_score_path)
            sanity_check_lab(lab_align)
            if compute_nosil_duration(lab_align, 0) < 1e-9:
                # skip
                print(
                    f"{splitext(name)[0]} is excluded from training due to too much silence."
                )
                black_list.append(splitext(name)[0])
                seg_idx += 1
                continue

            # Pau/sil lengths may differ in score and alignment, so remove it in case.
            lab_align = trim_sil_and_pau(lab_align)
            lab_score = trim_sil_and_pau(lab_score)

            # Extract note onsets and let's compute a offset
            note_indices = get_note_indices(lab_score)

            # offset = argmin_{b} \sum_{t=1}^{T}|x-(y+b)|^2
            # assuming there's a constant offset; tempo is same through the song
            onset_align = np.asarray(lab_align[note_indices].start_times)
            onset_score = np.asarray(lab_score[note_indices].start_times)

            # Offset adjustment
            segment_offset = (onset_align - onset_score).mean()
            segment_offset = int(round(segment_offset / 50000) * 50000)
            if apply_offset_correction:
                if config["global_offset_correction"]:
                    offset_ = global_offset
                else:
                    offset_ = segment_offset
                print(f"{name} offset (in sec): {offset_ * 1e-7}")
            else:
                offset_ = 0
            # apply
            lab_score.start_times = list(np.asarray(lab_score.start_times) + offset_)
            lab_score.end_times = list(np.asarray(lab_score.end_times) + offset_)
            onset_score += offset_

            # Exclude large diff parts (probably a bug of musicxml or alignment though)
            valid_note_indices = []
            for idx, (a, b) in enumerate(zip(onset_align, onset_score)):
                note_idx = note_indices[idx]
                lag = np.abs(a - b) / 50000
                if _is_silence(lab_score.contexts[note_idx]):
                    if lag >= float(
                        config["timelag_allowed_range_rest"][0]
                    ) and lag <= float(config["timelag_allowed_range_rest"][1]):
                        valid_note_indices.append(note_idx)
                else:
                    if lag >= float(
                        config["timelag_allowed_range"][0]
                    ) and lag <= float(config["timelag_allowed_range"][1]):
                        valid_note_indices.append(note_idx)

            if len(valid_note_indices) < len(note_indices):
                D = len(note_indices) - len(valid_note_indices)
                print(f"{utt_id}.lab: {D}/{len(note_indices)} time-lags are excluded.")
            if (
                len(valid_note_indices) < 2
                or len(valid_note_indices) < len(note_indices) / 2
            ):
                print(
                    f"{splitext(name)[0]} is excluded from training due to num(valid_note_indice)({len(valid_note_indices)}) < 2 or len(valid_note_indices){len(valid_note_indices)} < len(note_indices){len(note_indices)} / 2."
                )
                black_list.append(splitext(name)[0])

            # Note onsets as labels
            lab_align = lab_align[valid_note_indices]
            lab_score = lab_score[valid_note_indices]

            # Save lab files
            lab_align_dst_path = join(lab_align_dst_dir, name)
            with open(lab_align_dst_path, "w") as of:
                of.write(str(lab_align))

            lab_score_dst_path = join(lab_score_dst_dir, name)
            with open(lab_score_dst_path, "w") as of:
                of.write(str(lab_score))

            seg_idx += 1

    # Prepare data for duration models

    dst_dir = join(out_dir, "duration")
    lab_align_dst_dir = join(dst_dir, "label_phone_align")

    for d in [lab_align_dst_dir]:
        os.makedirs(d, exist_ok=True)

    print("Prepare data for duration models")
    full_lab_align_files = sorted(glob(join(full_align_dir, f"*{suffix}")))
    for lab_align_path in tqdm(full_lab_align_files):
        utt_id = splitext(basename(lab_align_path))[0]
        seg_idx = 0

        while True:
            lab_align_path = join(full_align_dir + "_seg", f"{utt_id}_seg{seg_idx}.lab")
            name = basename(lab_align_path)
            if splitext(name)[0] in black_list:
                seg_idx += 1
                continue
            assert seg_idx > 0 or exists(lab_align_path)
            if not exists(lab_align_path):
                break

            lab_align = hts.load(lab_align_path)
            sanity_check_lab(lab_align)
            lab_align = fix_offset(lab_align)
            if len(lab_align) < 2:
                print(
                    f"{splitext(name)[0]} is excluded from training due to too short LAB_length: {len(lab_align)}"
                )
                black_list.append(splitext(name)[0])

            # Save lab file
            lab_align_dst_path = join(lab_align_dst_dir, name)
            with open(lab_align_dst_path, "w") as of:
                of.write(str(lab_align))

            seg_idx += 1

    # Prepare data for acoustic models

    dst_dir = join(out_dir, "acoustic")
    wav_dst_dir = join(dst_dir, "wav")
    lab_align_dst_dir = join(dst_dir, "label_phone_align")
    lab_score_dst_dir = join(dst_dir, "label_phone_score")

    for d in [wav_dst_dir, lab_align_dst_dir, lab_score_dst_dir]:
        os.makedirs(d, exist_ok=True)

    print("Prepare data for acoustic models")
    full_lab_align_files = sorted(glob(join(full_align_dir, f"*{suffix}")))

    for lab_align_path in full_lab_align_files:
        # for lab_align_path, lab_score_path in zip(full_lab_align_files, full_lab_score_files):
        utt_id = splitext(basename(lab_align_path))[0]
        name = splitext(basename(lab_align_path))[0]
        spk, name = name.split("_")
        wav_path = join(args.jaCappella_root, spk, f"{name}.wav")
        assert wav_path
        wav, sr = librosa.load(wav_path, sr=48000)

        # if gain_normalize:
        #     wav = wav / wav.max() * 0.99

        seg_idx = 0
        while True:
            lab_align_path = join(full_align_dir + "_seg", f"{utt_id}_seg{seg_idx}.lab")
            lab_score_path = join(full_score_dir + "_seg", f"{utt_id}_seg{seg_idx}.lab")
            name = splitext(basename(lab_align_path))[0]
            if name in black_list:
                # skip
                seg_idx += 1
                continue
            assert seg_idx > 0 or exists(lab_align_path)
            if not exists(lab_align_path):
                break
            lab_align = hts.load(lab_align_path)
            lab_score = hts.load(lab_score_path)

            # Make a slice of audio and then save it
            b, e = int(lab_align[0][0] * 1e-7 * sr), int(lab_align[-1][1] * 1e-7 * sr)
            wav_silce = wav[b:e]
            wav_slice_path = join(wav_dst_dir, f"{name}.wav")
            # TODO: consider explicit subtype
            sf.write(wav_slice_path, wav_silce, sr)

            # Set the beginning time to be zero for convenience
            lab_align = fix_offset(lab_align)
            sanity_check_lab(lab_align)
            lab_score = fix_offset(lab_score)

            # Save label
            lab_align_dst_path = join(lab_align_dst_dir, f"{name}.lab")
            with open(lab_align_dst_path, "w") as of:
                of.write(str(lab_align))

            lab_score_dst_path = join(lab_score_dst_dir, f"{name}.lab")
            with open(lab_score_dst_path, "w") as of:
                of.write(str(lab_score))

            seg_idx += 1

print("Complete data preparation")
sys.exit(0)
