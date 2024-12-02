import argparse
import os
import sys

"""
## 既存のリストを元に,楽曲を変えずsegment情報だけ追記

python $NNSVS_ROOT/recipes/_common/db/jaCappella_multi/split_by_song.py \
    data/list/utt_list.txt data/list/$train_set.list \
        data/list/$dev_set.list \
            $eval_set.list

"""

# parse arguments


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ref_list_dir_path",
        type=str,
    )
    parser.add_argument(
        "utt_list_dir_path",
        type=str,
    )
    return parser


args = get_parser().parse_args(sys.argv[1:])
ref_list_dir_path = args.ref_list_dir_path
utt_list_dir_path = args.utt_list_dir_path


# 曲名リスト
# 前提: 音声ファイル名形式が speaker_songname_id.wav
def make_song_list(list_name):
    list_path = os.path.join(ref_list_dir_path, list_name + ".list")
    song_list = []
    with open(list_path, mode="r") as f:
        sample_list = f.readlines()
    for sample_name in sample_list:
        song_name = os.path.basename(sample_name).split("_")[1]
        song_list.append(song_name)

    song_list = sorted(list(set(song_list)))
    return song_list


dev_song_list = make_song_list("dev")
eval_song_list = make_song_list("eval")

with open(os.path.join(utt_list_dir_path, "utt_list.txt"), mode="r") as f:
    utt_list = f.readlines()

# 同一楽曲のサンプルは全て同じリストに入れる
# 楽曲レベルで均等にしているため，リストレベルではサンプル数の不均衡が生じる場合がある
with open(os.path.join(utt_list_dir_path, "train_no_dev.list"), mode="w") as train_f:
    with open(os.path.join(utt_list_dir_path, "dev.list"), mode="w") as dev_f:
        with open(os.path.join(utt_list_dir_path, "eval.list"), mode="w") as eval_f:
            for wav_name in utt_list:
                song_name = wav_name.split("_")[1]
                if song_name in dev_song_list:
                    dev_f.write(wav_name)
                elif song_name in eval_song_list:
                    eval_f.write(wav_name)
                else:
                    train_f.write(wav_name)
exit()
