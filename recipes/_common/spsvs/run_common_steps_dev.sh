# NOTE: This script is for development purpose. It is subject to change without any notice.
# NOTE: The script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: Feature generation"
  . $NNSVS_COMMON_ROOT/feature_generation.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Training time-lag model"
  . $NNSVS_COMMON_ROOT/train_timelag.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Training duration model"
  . $NNSVS_COMMON_ROOT/train_duration.sh
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Training acoustic model"
  . $NNSVS_COMMON_ROOT/train_acoustic.sh
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: save timing information"
    . $NNSVS_COMMON_ROOT/evaluate_timing.sh
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: Synthesis waveforms"
  . $NNSVS_COMMON_ROOT/synthesis.sh
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Prepare input/output features for post-filter"
  . $NNSVS_COMMON_ROOT/prepare_postfilter.sh
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: Training post-filter"
  . $NNSVS_COMMON_ROOT/train_postfilter.sh
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "stage 9: Prepare vocoder input/output features"
  . $NNSVS_COMMON_ROOT/prepare_voc_features.sh
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "stage 10: Training vocoder using parallel_wavegan"
  if [ ! -z ${pretrained_vocoder_checkpoint} ]; then
    extra_args="--resume $pretrained_vocoder_checkpoint"
  else
    extra_args=""
  fi
  # NOTE: copy normalization stats to expdir for convenience
  mkdir -p $expdir/$vocoder_model
  cp -v $dump_norm_dir/in_vocoder*.npy $expdir/$vocoder_model
  xrun parallel-wavegan-train --config conf/train_parallel_wavegan/${vocoder_model}.yaml \
    --train-dumpdir $dump_norm_dir/$train_set/in_vocoder \
    --dev-dumpdir $dump_norm_dir/$dev_set/in_vocoder/ \
    --outdir $expdir/$vocoder_model $extra_args
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "stage 11: Training uSFGAN vocoder"
  . $NNSVS_COMMON_ROOT/train_usfgan.sh
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "stage 12: Synthesize waveforms from exracted features"
  . $NNSVS_COMMON_ROOT/anasyn.sh
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "stage 13: Training SiFi-GAN vocoder"
  . $NNSVS_COMMON_ROOT/train_sifigan.sh
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
  echo "stage 1: Feature generation for Multi-track training"
  . $NNSVS_COMMON_ROOT/feature_generation_multitrack.sh
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
  echo "stage 2: Multi-track training time-lag model"
  . $NNSVS_COMMON_ROOT/train_timelag_multitrack.sh
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
  echo "stage 3: Multi-track training duration model"
  . $NNSVS_COMMON_ROOT/train_duration_multitrack.sh
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ]; then
  echo "stage 4: Multi-track training acoustic model"
  . $NNSVS_COMMON_ROOT/train_acoustic_multitrack.sh
fi

if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ]; then
  echo "stage 5: Calc and save timelag and duration"
  . $NNSVS_COMMON_ROOT/evaluate_timing_multitrack.sh
fi

if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ]; then
  echo "stage 6: Synthesis waveforms and generate acoustic features"
  . $NNSVS_COMMON_ROOT/synthesis_multitrack.sh
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
  echo "Pack models for SVS"
  # PWG
  if [[ -z "${vocoder_eval_checkpoint}" && -f ${expdir}/${vocoder_model}/config.yml ]]; then
    vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
  # uSFGAN
  elif [[ -z "${vocoder_eval_checkpoint}" && -f ${expdir}/${vocoder_model}/config.yaml ]]; then
    vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
  fi
  # Determine the directory name of a packed model
  if [ -e "$vocoder_eval_checkpoint" ]; then
    # PWG's expdir or packed model's dir
    voc_dir=$(dirname $vocoder_eval_checkpoint)
    # PWG's expdir
    if [ -e ${voc_dir}/config.yml ]; then
      voc_config=${voc_dir}/config.yml
      vocoder_config_name=$(basename $(grep config: ${voc_config} | awk '{print $2}'))
      vocoder_config_name=${vocoder_config_name/.yaml/}
    # uSFGAN
    elif [ -e ${voc_dir}/config.yaml ]; then
      voc_config=${voc_dir}/config.yaml
      vocoder_config_name=$(basename $(grep out_dir: ${voc_config} | awk '{print $2}'))
    # Packed model's dir
    elif [ -e ${voc_dir}/vocoder_model.yaml ]; then
      # NOTE: assuming PWG for now
      voc_config=${voc_dir}/vocoder_model.yaml
      vocoder_config_name=$(basename $(grep config: ${voc_config} | awk '{print $2}'))
      vocoder_config_name=${vocoder_config_name/.yaml/}
    else
      echo "ERROR: vocoder config is not found!"
      exit 1
    fi
    dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}_${vocoder_config_name}
  else
    dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}
  fi

  if [[ ${acoustic_features} == *"melf0"* ]]; then
    feature_type="melf0"
  else
    feature_type="world"
  fi

  # Find settings needed for inference
  local_config_path=conf/prepare_features/acoustic/${acoustic_features}.yaml
  global_config_path=$NNSVS_ROOT/nnsvs/bin/conf/prepare_features/acoustic/${acoustic_features}.yaml
  if [ -e $local_config_path ]; then
    relative_f0=$(grep relative_f0 $local_config_path | awk '{print $2}')
    subphone_features=$(grep subphone_features $local_config_path | awk '{print $2}')
    use_world_codec=$(grep use_world_codec $local_config_path | awk '{print $2}')
    frame_period=$(grep frame_period $local_config_path | awk '{print $2}')
  elif [ -e $global_config_path ]; then
    relative_f0=$(grep relative_f0 $global_config_path | awk '{print $2}')
    subphone_features=$(grep subphone_features $global_config_path | awk '{print $2}')
    use_world_codec=$(grep use_world_codec $global_config_path | awk '{print $2}')
    frame_period=$(grep frame_period $global_config_path | awk '{print $2}')
  else
    echo "config file not found: $local_config_path or $global_config_path"
    exit 1
  fi

  mkdir -p $dst_dir
  # global config file
  cat >${dst_dir}/config.yaml <<EOL
# Global configs
sample_rate: ${sample_rate}
frame_period: ${frame_period}
log_f0_conditioning: true
use_world_codec: ${use_world_codec}
feature_type: ${feature_type}

# Model-specific synthesis configs
timelag:
    allowed_range: [-20, 20]
    allowed_range_rest: [-40, 40]
    force_clip_input_features: true
duration:
    force_clip_input_features: true
acoustic:
    subphone_features: ${subphone_features}
    force_clip_input_features: true
    relative_f0: ${relative_f0}
EOL

  . $NNSVS_COMMON_ROOT/pack_model.sh
fi
