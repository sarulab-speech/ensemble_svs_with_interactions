from functools import partial
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import to_absolute_path
from nnmnkwii import metrics
from nnsvs.base import PredictionType
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu, mdn_loss
from nnsvs.multistream import split_streams
from nnsvs.train_util import (
    collate_fn_multitrack,
    collate_fn_random_segments,
    collate_fn_syncmultitrack,
    get_stream_weight,
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup,
)
from nnsvs.util import PyTorchStandardScaler, make_non_pad_mask
from omegaconf import DictConfig
from torch import nn
from torch.cuda.amp import autocast


@torch.no_grad()
def compute_distortions(pred_out_feats, out_feats, lengths, out_scaler):
    assert pred_out_feats.shape == out_feats.shape
    out_feats = out_scaler.inverse_transform(out_feats)
    pred_out_feats = out_scaler.inverse_transform(pred_out_feats)

    dist = {}
    try:
        dist["ObjEval_RMSE"] = np.sqrt(
            metrics.mean_squared_error(out_feats, pred_out_feats, lengths=lengths)
        )
    except ZeroDivisionError:
        pass

    return dist


def train_step(
    model,
    optimizer,
    grad_scaler,
    train,
    in_feats_list,  # ([batch, frames, dim], [batch, frames, dim])
    out_feats_list,
    spks_list,
    lengths_list,
    mask_list,
    out_scaler,
    feats_criterion="mse",
    stream_wise_loss=False,
    stream_weights=None,
    stream_sizes=None,
):
    model.train() if train else model.eval()
    optimizer.zero_grad()

    if feats_criterion in ["l2", "mse"]:
        criterion = nn.MSELoss(reduction="none")
    elif feats_criterion in ["l1", "mae"]:
        criterion = nn.L1Loss(reduction="none")
    else:
        raise RuntimeError("not supported criterion")

    prediction_type = (
        model.module.prediction_type()
        if isinstance(model, nn.DataParallel)
        else model.prediction_type()
    )
    in_feats0, in_feats1 = in_feats_list
    in_feats = torch.concat((in_feats0, in_feats1), dim=2)
    lengths = torch.max(lengths_list[0], lengths_list[1])

    # Apply preprocess if required (e.g., FIR filter for shallow AR)
    # defaults to no-op
    for idx in range(len(out_feats_list)):
        if isinstance(model, nn.DataParallel):
            out_feats_list[idx] = model.module.preprocess_target(out_feats_list[idx])
        else:
            out_feats_list[idx] = model.preprocess_target(out_feats_list[idx])
    # Run forward
    with autocast(enabled=grad_scaler is not None):
        pred_out_feats = model(in_feats, tuple(spks_list))

    # Mask (B, T, 1)
    len_mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
    mask = mask_list[0] if mask_list is not None else len_mask
    assert torch.all(mask.squeeze(-1) == mask_list[0])
    import pdb

    pdb.set_trace()
    # Compute loss
    if prediction_type == PredictionType.PROBABILISTIC:
        pi, sigma, mu = pred_out_feats
        # (B, max(T)) or (B, max(T), D_out)
        mask_ = mask if len(pi.shape) == 4 else mask.squeeze(-1)
        len_mask_ = len_mask if len(pi.shape) == 4 else len_mask.squeeze(-1)
        # Compute loss and apply mask
        assert (
            out_feats_list[0].shape[1] == pi.shape[1] and pi.shape[1] == mask.shape[1]
        )
        with autocast(enabled=grad_scaler is not None):
            loss = mdn_loss(pi, sigma, mu, out_feats_list[0], reduce=False)
        loss = loss.masked_select(mask_).mean()
    else:
        if stream_wise_loss:
            w = get_stream_weight(stream_weights, stream_sizes).to(in_feats.device)
            streams = split_streams(out_feats_list[0], stream_sizes)
            pred_streams = split_streams(pred_out_feats, stream_sizes)
            loss = 0
            for pred_stream, stream, sw in zip(pred_streams, streams, w):
                with autocast(enabled=grad_scaler is not None):
                    loss += (
                        sw
                        * criterion(
                            pred_stream.masked_select(mask), stream.masked_select(mask)
                        ).mean()
                    )
        else:
            with autocast(enabled=grad_scaler is not None):
                loss = criterion(
                    pred_out_feats.masked_select(mask),
                    out_feats_list[0].masked_select(mask),
                ).mean()

    if prediction_type == PredictionType.PROBABILISTIC:
        with torch.no_grad():
            pred_out_feats_ = mdn_get_most_probable_sigma_and_mu(pi, sigma, mu)[1]
    else:
        pred_out_feats_ = pred_out_feats
    distortions = compute_distortions(
        pred_out_feats_, out_feats_list[0], lengths, out_scaler
    )

    # individual loss
    if train:
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # TODO interaction loss calc

    return loss, distortions


def train_loop(
    config,
    logger,
    device,
    model,
    optimizer,
    lr_scheduler,
    grad_scaler,
    data_loaders,
    samplers,
    writer,
    out_scaler,
    use_mlflow,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_dev_loss = torch.finfo(torch.float32).max
    last_dev_loss = torch.finfo(torch.float32).max

    if "feats_criterion" not in config.train:
        logger.warning(
            "feats_criterion is not found in the train config. Fallback to MSE."
        )
        feats_criterion = "mse"
    else:
        feats_criterion = config.train.feats_criterion

    if dist.is_initialized() and dist.get_rank() != 0:

        def tqdm(x, **kwargs):
            return x

    else:
        from tqdm import tqdm

    train_iter = 1
    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if dist.is_initialized() and train and samplers[phase] is not None:
                samplers[phase].set_epoch(epoch)
            running_loss = 0
            running_metrics = {}
            for data in tqdm(data_loaders[phase], desc=f"{phase} iter", leave=False):
                if config.data.padding == "left":
                    (
                        in_feats0,
                        out_feats0,
                        spks0,
                        lengths0,
                        in_feats1,
                        out_feats1,
                        spks1,
                        lengths1,
                    ) = data
                elif config.data.padding == "time":
                    (
                        in_feats0,
                        out_feats0,
                        spks0,
                        lengths0,
                        mask0,
                        in_feats1,
                        out_feats1,
                        spks1,
                        lengths1,
                        mask1,
                    ) = data

                # NOTE: This is needed for pytorch's PackedSequence
                lengths0, indices0 = torch.sort(lengths0, dim=0, descending=True)
                lengths1, indices1 = torch.sort(lengths1, dim=0, descending=True)
                in_feats0, out_feats0, spks0 = (
                    in_feats0[indices0].to(device),
                    out_feats0[indices0].to(device),
                    spks0[indices0].to(device, dtype=torch.int),
                )
                in_feats1, out_feats1, spks1 = (
                    in_feats1[indices1].to(device),
                    out_feats1[indices1].to(device),
                    spks1[indices1].to(device, dtype=torch.int),
                )
                if config.data.padding == "left":
                    mask_list = None
                elif config.data.padding == "time":
                    mask0 = mask0[indices0].to(device, dtype=torch.bool)
                    mask1 = mask1[indices1].to(device, dtype=torch.bool)
                    mask_list = (mask0, mask1)

                loss, log_metrics = train_step(
                    model=model,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    train=train,
                    in_feats_list=(in_feats0, in_feats1),
                    out_feats_list=[out_feats0, out_feats1],
                    spks_list=(spks0, spks1),
                    lengths_list=(lengths0, lengths1),
                    mask_list=mask_list,
                    out_scaler=out_scaler,
                    feats_criterion=feats_criterion,
                    stream_wise_loss=config.train.stream_wise_loss,
                    stream_weights=config.model.stream_weights,
                    stream_sizes=config.model.stream_sizes,
                )

                if train:
                    if writer is not None:
                        for key, val in log_metrics.items():
                            writer.add_scalar(f"{key}_Step/{phase}", val, train_iter)
                    train_iter += 1

                running_loss += loss.item()
                for k, v in log_metrics.items():
                    try:
                        running_metrics[k] += float(v)
                    except KeyError:
                        running_metrics[k] = float(v)

            ave_loss = running_loss / len(data_loaders[phase])
            if writer is not None:
                writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)
            if use_mlflow:
                mlflow.log_metric(f"{phase}_loss", ave_loss, step=epoch)

            ave_loss = running_loss / len(data_loaders[phase])
            logger.info("[%s] [Epoch %s]: loss %s", phase, epoch, ave_loss)
            if writer is not None:
                writer.add_scalar(f"Loss_Epoch/{phase}", ave_loss, epoch)
            if use_mlflow:
                mlflow.log_metric(f"{phase}_loss", ave_loss, step=epoch)

            for k, v in running_metrics.items():
                ave_v = v / len(data_loaders[phase])
                if writer is not None:
                    writer.add_scalar(f"{k}_Epoch/{phase}", ave_v, epoch)
                if use_mlflow:
                    mlflow.log_metric(f"{phase}_{k}", ave_v, step=epoch)

            if not train:
                last_dev_loss = ave_loss
            if not train and ave_loss < best_dev_loss:
                best_dev_loss = ave_loss
                save_checkpoint(
                    logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=True
                )

        lr_scheduler.step()
        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(
                logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=False
            )

    save_checkpoint(
        logger, out_dir, model, optimizer, lr_scheduler, config.train.nepochs
    )
    logger.info("The best loss was %s", best_dev_loss)
    if use_mlflow:
        mlflow.log_metric("best_dev_loss", best_dev_loss, step=epoch)
        mlflow.log_artifacts(out_dir)

    return last_dev_loss


@hydra.main(config_path="conf/train", config_name="config")
def my_app(config: DictConfig) -> None:
    if "max_time_frames" in config.data and config.data.max_time_frames > 0:
        assert (False, "this config is not allowed in multi-track training")
        collate_fn = partial(
            collate_fn_random_segments, max_time_frames=config.data.max_time_frames
        )
    else:
        if config.data.padding == "time":
            collate_fn = collate_fn_syncmultitrack
        elif config.data.padding == "left":
            collate_fn = collate_fn_multitrack
        else:
            assert False

    if config.train.use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        optimizer,
        lr_scheduler,
        grad_scaler,
        data_loaders,
        samplers,
        writer,
        logger,
        _,
        out_scaler,
    ) = setup(config, device, collate_fn, multitrack=True)

    out_scaler = PyTorchStandardScaler(
        torch.from_numpy(out_scaler.mean_), torch.from_numpy(out_scaler.scale_)
    ).to(device)
    use_mlflow = config.mlflow.enabled

    if use_mlflow:
        with mlflow.start_run() as run:
            # NOTE: modify out_dir when running with mlflow
            config.train.out_dir = f"{config.train.out_dir}/{run.info.run_id}"
            save_configs(config)
            log_params_from_omegaconf_dict(config)
            last_dev_loss = train_loop(
                config=config,
                logger=logger,
                device=device,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_scaler=grad_scaler,
                data_loaders=data_loaders,
                samplers=samplers,
                writer=writer,
                out_scaler=out_scaler,
                use_mlflow=use_mlflow,
            )
    else:
        save_configs(config)
        last_dev_loss = train_loop(
            config=config,
            logger=logger,
            device=device,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            data_loaders=data_loaders,
            samplers=samplers,
            writer=writer,
            out_scaler=out_scaler,
            use_mlflow=use_mlflow,
        )

    return last_dev_loss


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
