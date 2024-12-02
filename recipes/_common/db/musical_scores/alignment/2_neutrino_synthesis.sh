# #!/bin/bash

# load YAML parser
. yaml_parser.sh
eval $(parse_yaml config.yaml "config_")

in_label_dir=$config_path_data_label
out_wav_dir=$config_path_data_synth_wav

python=$config_path_python
neutrino=$config_path_neutrino # NEUTRINO path
voice=$config_neutrino_voice

genres=(${config_song_genres// / })
song_parts=(${config_song_parts// / })

for genre in ${genres[@]}; do
  for song in `ls -d ${in_label_dir}/$genre/full/*_parts`; do
    song_name=`basename $song | sed "s/_parts$//g"`
    for song_part in ${song_parts[@]}; do
      label_path="$song/$song_name-${song_part}.lab"
      $python scripts/neutrino_waveform_synthesis.py \
        --neutrino $neutrino \
        --output $out_wav_dir/$genre \
        --voice $voice \
        $label_path
    done
  done
done

