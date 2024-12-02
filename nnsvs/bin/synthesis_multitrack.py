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

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(config.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # NOTE: this is used for GV post-filtering
    acoustic_out_static_scaler = extract_static_scaler(
        acoustic_out_scaler, acoustic_config
    )

    # Vocoder
    if config.vocoder.checkpoint is not None and len(config.vocoder.checkpoint) > 0:
        vocoder, vocoder_in_scaler, vocoder_config = load_vocoder(
            to_absolute_path(config.vocoder.checkpoint),
            device,
            acoustic_config,
        )
    else:
        vocoder, vocoder_in_scaler, vocoder_config = None, None, None
        if config.synthesis.vocoder_type != "world":
            logger.warning("Vocoder checkpoint is not specified")
            logger.info(f"Use world instead of {config.synthesis.vocoder_type}.")
        config.synthesis.vocoder_type = "world"

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

            init_seed(1234)
            spk_list = config.spk_list
            if config.spk_name == "Multi":
                spk_0 = utt_id_0.split("_")[0]

                # Comment out if you check ritsu
                # if spk_0 == "ritsu":
                #     continue

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

                fs = ["timelag", "duration"]
                out_path = join(out_dir, "timelag", f"{utt_id_0}_with_{utt_id_1}.npy")
                np.save(out_path, lag_0)
                out_path = join(out_dir, "duration", f"{utt_id_0}_with_{utt_id_1}.npy")
                np.save(out_path, durations_0)

            # Predict acoustic features
            acoustic_features = predict_acoustic_multitrack(
                device=device,
                labels_list=[duration_modified_labels_0, duration_modified_labels_1],
                spks_list=[spkid_0, spkid_1],
                acoustic_model=acoustic_model,
                acoustic_config=acoustic_config,
                acoustic_in_scaler=acoustic_in_scaler,
                acoustic_out_scaler=acoustic_out_scaler,
                binary_dict=binary_dict,
                numeric_dict=numeric_dict,
                subphone_features=config.synthesis.subphone_features,
                log_f0_conditioning=config.synthesis.log_f0_conditioning,
                force_clip_input_features=config.acoustic.force_clip_input_features,
                f0_shift_in_cent=config.synthesis.pre_f0_shift_in_cent,
            )

            # NOTE: the output of this function is tuple of features
            # e.g., (mgc, lf0, vuv, bap)
            multistream_features = postprocess_acoustic(
                device=device,
                acoustic_features=acoustic_features,
                duration_modified_labels=duration_modified_labels_0,
                binary_dict=binary_dict,
                numeric_dict=numeric_dict,
                acoustic_config=acoustic_config,
                acoustic_out_static_scaler=acoustic_out_static_scaler,
                postfilter_model=None,  # NOTE: learned post-filter is not supported
                postfilter_config=None,
                postfilter_out_scaler=None,
                sample_rate=config.synthesis.sample_rate,
                frame_period=config.synthesis.frame_period,
                relative_f0=config.synthesis.relative_f0,
                feature_type=config.synthesis.feature_type,
                post_filter_type=config.synthesis.post_filter_type,
                trajectory_smoothing=config.synthesis.trajectory_smoothing,
                trajectory_smoothing_cutoff=config.synthesis.trajectory_smoothing_cutoff,
                trajectory_smoothing_cutoff_f0=config.synthesis.trajectory_smoothing_cutoff_f0,
                vuv_threshold=config.synthesis.vuv_threshold,
                f0_shift_in_cent=config.synthesis.post_f0_shift_in_cent,
                vibrato_scale=1.0,
                force_fix_vuv=config.synthesis.force_fix_vuv,
            )

            # Generate waveform by vocoder
            wav = predict_waveform(
                device=device,
                multistream_features=multistream_features,
                vocoder=vocoder,
                vocoder_config=vocoder_config,
                vocoder_in_scaler=vocoder_in_scaler,
                sample_rate=config.synthesis.sample_rate,
                frame_period=config.synthesis.frame_period,
                use_world_codec=config.synthesis.use_world_codec,
                feature_type=config.synthesis.feature_type,
                vocoder_type=config.synthesis.vocoder_type,
                vuv_threshold=config.synthesis.vuv_threshold,
            )

            wav = postprocess_waveform(
                wav=wav,
                sample_rate=config.synthesis.sample_rate,
                dtype=np.int16,
                peak_norm=False,
                loudness_norm=False,
            )

            out_wav_path = join(out_dir, f"{utt_id_0}_with_{utt_id_1}.wav")
            wavfile.write(
                out_wav_path,
                rate=config.synthesis.sample_rate,
                data=wav
                # out_wav_path, rate=config.synthesis.sample_rate, data=wav.astype(np.int16)
            )

            # out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
            # out_scaler = StandardScaler(
            #     out_scaler.mean_, out_scaler.var_, out_scaler.scale_
            # )
            # # multistream_features = out_scaler.inverse_transform(np.concatenate(multistream_features, axis=1))
            # multistream_features = multistream_features[:,0:60],multistream_features[:,60:61],multistream_features[:,61:62],multistream_features[:,62:67],
            fs = ["mgc", "logF0", "vuv", "bap"]
            for i, feature_name in enumerate(fs):
                out_path = join(
                    out_dir, feature_name, f"{utt_id_0}_with_{utt_id_1}.npy"
                )
                np.save(out_path, multistream_features[i])


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
