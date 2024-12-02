#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

function xrun() {
  set -x
  $@
  set +x
}

script_dir=$(
  cd $(dirname ${BASH_SOURCE:-$0})
  pwd
)
NNSVS_ROOT=$script_dir/../../../
NNSVS_COMMON_ROOT=$NNSVS_ROOT/recipes/_common/spsvs
. $NNSVS_ROOT/utils/yaml_parser.sh || exit 1
config_path=$script_dir/config.yaml

eval $(parse_yaml "./config.yaml" "")

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

dumpdir=dump
dumpdir_multitrack=dump_multitrack

dump_org_dir=$dumpdir/$spk/org
dump_norm_dir=$dumpdir/$spk/norm
dump_org_dir_multitrack=$dumpdir_multitrack/$spk/org
dump_norm_dir_multitrack=$dumpdir_multitrack/$spk/norm

stage=0
stop_stage=0

. $NNSVS_ROOT/utils/parse_options.sh || exit 1

# exp name
if [ -z ${tag:=} ]; then
  expname=${spk}
else
  expname=${spk}_${tag}
fi
expdir=exp/$expname
expdir_multitrack=exp_multitrack/$expname

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
  echo "stage 0: Data preparation for multi-track training"
  # the following three directories will be created
  # 1) data/timelag 2) data/duration 3) data/acoustic

  # Split the audio data into segments and save.
  python $NNSVS_ROOT/recipes/_common/db/jaCappella_multi/data_prep_multitrack.py $db_root_multi_track $out_dir_multi_track $config_path

  echo "train/dev/eval split"
  mkdir -p $out_dir_multi_track/list
  # List up the segment names.
  find $out_dir_multi_track/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; |
    sort >$out_dir_multi_track/list/utt_list.txt

  # Split the dataset. Segments of the same music are divided into the same split.
  python $NNSVS_ROOT/recipes/_common/db/jaCappella_multi/split_by_song.py $out_dir_multi_track/list/utt_list.txt $out_dir_multi_track/list/$train_set.list $out_dir_multi_track/list/$dev_set.list $out_dir_multi_track/list/$eval_set.list
fi

# Run the rest of the steps
# Please check the script file for more details
. $NNSVS_COMMON_ROOT/run_common_steps_dev.sh
