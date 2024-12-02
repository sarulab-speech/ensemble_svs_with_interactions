# #!/bin/bash

# load YAML parser
. yaml_parser.sh
eval $(parse_yaml config.yaml "config_")

synth_wav_dir=$config_path_data_synth_wav
recorded_wav_dir=$config_path_data_recorded_wav
out_label_dir=$config_path_data_align

python=$config_path_python

genres=(${config_song_genres// / })
song_parts=(${config_song_parts// / })

for genre in ${genres[@]}; do
  for song in `ls -d ${synth_wav_dir}/$genre/*_parts`; do
    song_name=`basename $song | sed "s/_parts$//g"`
    for song_part in ${song_parts[@]}; do
      src_timing_path="$song/$song_name-$song_part.lab"
      src_wav_path="$song/$song_name-$song_part.wav"

      # 一時的な処置
      # song_part=`echo $song_part | sed 's/lead_vocals1/lead_vocal1/g'`

      recorded_wav_path="$recorded_wav_dir/$genre/$song_name/$song_part.wav"
      out_label_path="$out_label_dir/$genre/${song_name}_parts/$song_name-$song_part.lab"

      $python scripts/label_alignment.py \
        $src_timing_path \
        $src_wav_path \
        $recorded_wav_path \
        $out_label_path
    done
  done
done
