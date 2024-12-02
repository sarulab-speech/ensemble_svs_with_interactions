import argparse
import glob
import os
import random
import sys

import matplotlib.pyplot as plt

"""
    python $NNSVS_ROOT/recipes/_common/db/jaCappella_multi/split_by_song.py data/list/utt_list.txt data/list/$train_set.list data/list/$dev_set.list $eval_set.list

"""

# parse arguments
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "utt_list_path",
        type=str,
    )
    parser.add_argument(
        "train_path",
        type=str,
    )
    parser.add_argument(
        "dev_path",
        type=str,
    )
    parser.add_argument(
        "eval_path",
        type=str,
    )
    return parser


args = get_parser().parse_args(sys.argv[1:])
utt_list_path = args.utt_list_path
train_path = args.train_path
dev_path = args.dev_path
eval_path = args.eval_path


with open(utt_list_path) as f:
    utt_list = f.readlines()

# 曲名リスト
# 前提: 音声ファイル名形式が speaker_songname_id.wav
song_list = list(utt_list)
for i, wav_name in enumerate(utt_list):
    _, song, _ = wav_name.split("_")
    song_list[i] = song
song_list = list(set(song_list))
random.shuffle(song_list)


# 8:1:1 train:dev:eval
eval_num = len(song_list) // 10
dev_set = set(song_list[0:eval_num])
eval_set = set(song_list[eval_num : eval_num * 2])
train_list = [x for x in song_list if (x not in dev_set) and (x not in eval_set)]
dev_list = [x for x in song_list if (x in dev_set)]
eval_list = [x for x in song_list if (x in eval_set)]


# 同一楽曲のサンプルは全て同じリストに入れる
# 楽曲レベルで均等にしているため，リストレベルではサンプル数の不均衡が生じる場合がある
with open(train_path, mode="w") as train_f:
    with open(dev_path, mode="w") as dev_f:
        with open(eval_path, mode="w") as eval_f:
            for wav_name in utt_list:
                song_name = wav_name.split("_")[1]
                if song_name in dev_list:
                    dev_f.write(wav_name)
                elif song_name in eval_list:
                    eval_f.write(wav_name)
                else:
                    train_f.write(wav_name)
