import os
from os.path import join

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from nnsvs.gen import (
    postprocess_acoustic,
    postprocess_waveform,
    predict_acoustic,
    predict_timing,
    predict_waveform,
)
from nnsvs.logger import getLogger
from nnsvs.util import (
    StandardScaler,
    extract_static_scaler,
    init_seed,
    load_utt_list,
    load_vocoder,
)
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm.auto import tqdm


def trim_sil_and_pau(lab, return_indices=False):
    forward = 0
    while "-sil" in lab.contexts[forward] or "-pau" in lab.contexts[forward]:
        forward += 1

    backward = len(lab) - 1
    while "-sil" in lab.contexts[backward] or "-pau" in lab.contexts[backward]:
        backward -= 1

    if return_indices:
        return lab[forward : backward + 1], forward, backward
    else:
        return lab[forward : backward + 1]


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # timelag
    timelag_config = OmegaConf.load(to_absolute_path(config.timelag.model_yaml))
    timelag_model = hydra.utils.instantiate(timelag_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.timelag.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    timelag_model.load_state_dict(checkpoint["state_dict"])
    timelag_in_scaler = joblib.load(to_absolute_path(config.timelag.in_scaler_path))
    timelag_out_scaler = joblib.load(to_absolute_path(config.timelag.out_scaler_path))
    timelag_model.eval()

    # duration
    duration_config = OmegaConf.load(to_absolute_path(config.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.duration.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    duration_model.load_state_dict(checkpoint["state_dict"])
    duration_in_scaler = joblib.load(to_absolute_path(config.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(config.duration.out_scaler_path))
    duration_model.eval()

    # Run synthesis for each utt.
    binary_dict, numeric_dict = hts.load_question_set(
        to_absolute_path(config.synthesis.qst)
    )

    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mgc"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logF0"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bap"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vuv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "timelag"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "duration"), exist_ok=True)
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    logger.info("Processes %s utterances...", len(utt_ids))
    for utt_id in tqdm(utt_ids):
        labels = hts.load(join(in_dir, f"{utt_id}.lab"))
        hts_frame_shift = int(config.synthesis.frame_period * 1e4)
        labels.frame_shift = hts_frame_shift
        init_seed(1234)
        spk_list = config.spk_list
        if config.spk_name == "Multi":
            spk = utt_id.split("_")[0]
            try:
                spkid = torch.IntTensor([spk_list.index(spk)]).to(device)
            except ValueError:
                print(f"{spk} is not in spk_list")
        else:
            spkid = None

        if config.synthesis.ground_truth_duration:
            duration_modified_labels = labels
        else:
            duration_modified_labels, lag, durations = predict_timing(
                device=device,
                labels=labels,
                spk=spkid,
                binary_dict=binary_dict,
                numeric_dict=numeric_dict,
                timelag_model=timelag_model,
                timelag_config=timelag_config,
                timelag_in_scaler=timelag_in_scaler,
                timelag_out_scaler=timelag_out_scaler,
                duration_model=duration_model,
                duration_config=duration_config,
                duration_in_scaler=duration_in_scaler,
                duration_out_scaler=duration_out_scaler,
                log_f0_conditioning=config.synthesis.log_f0_conditioning,
                allowed_range=config.timelag.allowed_range,
                allowed_range_rest=config.timelag.allowed_range_rest,
                force_clip_input_features=config.timelag.force_clip_input_features,
                frame_period=config.synthesis.frame_period,
            )
            _, a, b = trim_sil_and_pau(labels, return_indices=True)
            # if utt_id_0 == "Vo1_nanatsunoko_seg2":
            #     import pdb;pdb.set_trace()

            # len(lab)-1-b 個のpauが末尾にある
            b = len(labels) - 1 - b
            lag = lag[a : len(lag) - b]

            fs = ["timelag", "duration"]
            out_path = join(out_dir, "timelag", f"{utt_id}.npy")
            np.save(out_path, lag)
            out_path = join(out_dir, "duration", f"{utt_id}.npy")
            np.save(out_path, durations)


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
