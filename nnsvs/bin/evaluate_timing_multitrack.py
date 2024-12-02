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
    predict_acoustic_multitrack,
    predict_timing_multitrack,
    predict_waveform,
)
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
    os.makedirs(os.path.join(out_dir, "timelag"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "duration"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "timelag_wo_trimming"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "timelag_mask"), exist_ok=True)
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    hts_frame_shift = int(config.synthesis.frame_period * 1e4)
    for utt_id_0 in tqdm(utt_ids):
        seg_id_0 = utt_id_0.split("_")[1:]
        for utt_id_1 in tqdm(utt_ids):
            seg_id_1 = utt_id_1.split("_")[1:]
            if seg_id_0 != seg_id_1:
                continue

            labels_0 = hts.load(join(in_dir, f"{utt_id_0}.lab"))
            labels_1 = hts.load(join(in_dir, f"{utt_id_1}.lab"))
            labels_0.frame_shift = hts_frame_shift
            labels_1.frame_shift = hts_frame_shift
            print(
                utt_id_0,
                utt_id_1,
                join(in_dir, f"{utt_id_0}.lab"),
                join(in_dir, f"{utt_id_1}.lab"),
            )
            init_seed(1234)
            spk_list = config.spk_list
            if config.spk_name == "Multi":
                spk_0 = utt_id_0.split("_")[0]
                if spk_0 == "ritsu":  # ritsu has no pair singer, so skip
                    continue
                spk_1 = utt_id_1.split("_")[0]
                try:
                    spkid_0 = torch.IntTensor([spk_list.index(spk_0)]).to(device)
                    spkid_1 = torch.IntTensor([spk_list.index(spk_1)]).to(device)
                except ValueError:
                    print(f"{spk_0} or {spk_1} are not in spk_list")
            else:
                spkid = None

            if config.synthesis.ground_truth_duration:
                duration_modified_labels_0 = labels_0
                duration_modified_labels_1 = labels_1
            else:
                (
                    duration_modified_labels_0,
                    lag_0,
                    durations_0,
                    mask_0,
                ) = predict_timing_multitrack(
                    device=device,
                    labels_list=[labels_0, labels_1],
                    spks_list=[spkid_0, spkid_1],
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
                (
                    duration_modified_labels_1,
                    lag_1,
                    durations_1,
                    mask_1,
                ) = predict_timing_multitrack(
                    device=device,
                    labels_list=[labels_1, labels_0],
                    spks_list=[spkid_1, spkid_0],
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

                if spk_0 == spk_1:
                    continue
                print(utt_id_0, utt_id_1)
                print(lag_0.shape, mask_0)

                _, a, b = trim_sil_and_pau(labels_0, return_indices=True)

                # len(lab)-1-b 個のpauが末尾にある
                b = len(labels_0) - 1 - b
                trimmed_lag_0 = lag_0[a : len(lag_0) - b]
                mask_0[:a] = False
                mask_0[len(lag_0) - b :] = False

                out_path = join(out_dir, "timelag", f"{utt_id_0}_with_{utt_id_1}.npy")
                np.save(out_path, trimmed_lag_0)

                out_path = join(out_dir, "duration", f"{utt_id_0}_with_{utt_id_1}.npy")
                np.save(out_path, durations_0)

                out_path = join(
                    out_dir, "timelag_wo_trimming", f"{utt_id_0}_with_{utt_id_1}.npy"
                )
                np.save(out_path, lag_0)

                out_path = join(
                    out_dir, "timelag_mask", f"{utt_id_0}_with_{utt_id_1}.npy"
                )
                np.save(out_path, mask_0)


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
