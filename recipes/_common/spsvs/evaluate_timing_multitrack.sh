# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/synthesis ]; then
    ext="--config-dir conf/synthesis"
else
    ext=""
fi

if [ -z $timelag_eval_checkpoint ]; then
    timelag_eval_checkpoint=best_loss.pth
fi
if [ -z $duration_eval_checkpoint ]; then
    duration_eval_checkpoint=best_loss.pth
fi


dst_name=timings_${timelag_model}_${duration_model}


for s in ${testsets[@]}; do
    for input in label_phone_score; do
        if [ $input = label_phone_score ]; then
            ground_truth_duration=false
        else
            ground_truth_duration=true
        fi
        
        xrun python $NNSVS_ROOT/nnsvs/bin/evaluate_timing_multitrack.py $ext \
            synthesis=$synthesis \
            synthesis.sample_rate=$sample_rate \
            synthesis.qst=$question_path \
            synthesis.ground_truth_duration=$ground_truth_duration \
            timelag.checkpoint=$expdir_multitrack/${timelag_model}/$timelag_eval_checkpoint \
            timelag.in_scaler_path=$dump_norm_dir_multitrack/in_timelag_scaler.joblib \
            timelag.out_scaler_path=$dump_norm_dir_multitrack/out_timelag_scaler.joblib \
            timelag.model_yaml=$expdir_multitrack/${timelag_model}/model.yaml \
            duration.checkpoint=$expdir_multitrack/${duration_model}/$duration_eval_checkpoint \
            duration.in_scaler_path=$dump_norm_dir_multitrack/in_duration_scaler.joblib \
            duration.out_scaler_path=$dump_norm_dir_multitrack/out_duration_scaler.joblib \
            duration.model_yaml=$expdir_multitrack/${duration_model}/model.yaml \
            utt_list=./data_multitrack/list/$s.list \
            in_dir=data_multitrack/acoustic/$input/ \
            out_dir=$expdir_multitrack/$dst_name/$s/$input/ \
            spk_name=${spk} \
            spk_list=${spk_list}
    done
done
