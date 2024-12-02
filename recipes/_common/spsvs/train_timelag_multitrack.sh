# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/train ]; then
    ext="--config-dir conf/train/timelag"
else
    ext=""
fi

if [ ! -z "${pretrained_expdir_for_multitrack}" ]; then
    resume_checkpoint=$pretrained_expdir_for_multitrack/${timelag_model}/latest.pth
else
    resume_checkpoint=
fi

# Hyperparameter search with Hydra + optuna
# mlflow is used to log the results of the hyperparameter search
if [[ ${timelag_hydra_optuna_sweeper_args+x} && ! -z $timelag_hydra_optuna_sweeper_args ]]; then
    hydra_opt="-m ${timelag_hydra_optuna_sweeper_args}"
    post_args="mlflow.enabled=true mlflow.experiment=${expname}_${timelag_model} hydra.sweeper.n_trials=${timelag_hydra_optuna_sweeper_n_trials}"
else
    hydra_opt=""
    post_args=""
fi

xrun python $NNSVS_ROOT/nnsvs/bin/train_multitrack.py $ext $hydra_opt \
    model=$timelag_model train=$timelag_train data=$timelag_data \
    data.train_no_dev.in_dir=$dump_norm_dir_multitrack/$train_set/in_timelag/ \
    data.train_no_dev.out_dir=$dump_norm_dir_multitrack/$train_set/out_timelag/ \
    data.dev.in_dir=$dump_norm_dir_multitrack/$dev_set/in_timelag/ \
    data.dev.out_dir=$dump_norm_dir_multitrack/$dev_set/out_timelag/ \
    data.in_scaler_path=$dump_norm_dir_multitrack/in_timelag_scaler.joblib \
    data.out_scaler_path=$dump_norm_dir_multitrack/out_timelag_scaler.joblib \
    train.out_dir=$expdir_multitrack/${timelag_model} \
    train.log_dir=tensorboard_multitrack/${expname}_${timelag_model} \
    train.resume.checkpoint=$resume_checkpoint $post_args
