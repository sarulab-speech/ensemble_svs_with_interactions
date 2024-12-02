# #!/bin/bash

# load YAML parser
. yaml_parser.sh
eval $(parse_yaml config.yaml "config_")

# set variables
in_xml_dir=$config_path_data_xml_full # dirname of musicXML files
out_xml_dir=$config_path_data_xml_parts
ext=$config_song_extension # extension of musicXML file
python=$config_path_python

genres=(${config_song_genres// / })

# run
for genre in ${genres[@]}; do
  echo $genre
  for xml_path in `ls ${in_xml_dir}/$genre/*.${ext}`; do
    # xml_path: ../scores_for_SVS/popular/茶摘.musicxml
    $python scripts/separate_parts_from_musicxml.py \
      --inputs $xml_path \
      --outdir $out_xml_dir/$genre
  done
  for xml_path in `ls ${out_xml_dir}/${genre}/*/*.${ext}`; do
    $python scripts/add_stretchs_for_melisma_lyrics.py ${xml_path}
  done
done
