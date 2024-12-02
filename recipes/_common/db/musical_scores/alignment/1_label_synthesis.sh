#!/bin/bash

# load YAML parser
. yaml_parser.sh
eval $(parse_yaml config.yaml "config_")

# set variables
xml_dir=$config_path_data_xml_parts # dirname of musicXML files
dic_dir=$config_neutrino_dic # dictionary 
label_dir=$config_path_data_label # dirname to save label files
ext=$config_song_extension  # extension of musicXML file

python=$config_path_python
neutrino=$config_path_neutrino # NEUTRINO path

genres=(${config_song_genres// / })
song_parts=(${config_song_parts// / })

# # copy setting
cp $dic_dir/japanese.utf_8.conf $neutrino/settings/dic/japanese.utf_8.conf
# nkf --sjis $dic_dir/japanese.utf_8.conf > $neutrino/settings/dic/japanese.shift_jis.conf
# nkf --euc $dic_dir/japanese.utf_8.conf > $neutrino/settings/dic/japanese.euc_jp.conf

cp $dic_dir/japanese.utf_8.table $neutrino/settings/dic/japanese.utf_8.table
# nkf --sjis $dic_dir/japanese.utf_8.table > $neutrino/settings/dic/japanese.shift_jis.table
# nkf --euc $dic_dir/japanese.utf_8.table > $neutrino/settings/dic/japanese.euc_jp.table

# run
for genre in ${genres[@]}; do
  for song in `ls -d ${xml_dir}/$genre/*_parts`; do
    song_name=`basename $song | sed "s/_parts$//g"`
    for song_part in ${song_parts[@]}; do
      xml_path="$song/$song_name-${song_part}.${ext}"
      $python scripts/label_synthesis.py \
        --neutrino ${neutrino} \
        --output ${label_dir}/$genre \
        ${xml_path}
    done
  done
done
