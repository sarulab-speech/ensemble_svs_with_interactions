import torch
from nnsvs.acoustic_models.util import pad_inference, pad_inference_multitrack
from nnsvs.base import BaseModel, PredictionType
from nnsvs.multistream import split_streams
from torch import nn

__all__ = [
    "MultistreamSeparateF0ParametricModel",
    "MultiSpeakerMultistreamSeparateF0ParametricModel",
    "NPSSMultistreamParametricModel",
    "NPSSMDNMultistreamParametricModel",
    "MultiTrackNPSSMDNMultistreamParametricModel",
    "V2MultiTrackNPSSMDNMultistreamParametricModel",
    "MultistreamSeparateF0MelModel",
    "MDNMultistreamSeparateF0MelModel",
]


class MultistreamSeparateF0ParametricModel(BaseModel):
    """Multi-stream model with a separate F0 prediction model

    acoustic features: [MGC, LF0, VUV, BAP]

    vib_model and vib_flags_model are optional and will be likely to be removed.

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mgc_model (nn.Module): MGC prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        bap_model (nn.Module): BAP prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        lf0_teacher_forcing (bool): Whether to use teacher forcing for F0 prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        encoder: nn.Module,
        mgc_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        bap_model: nn.Module,
        vib_model: nn.Module = None,  # kept as is for compatibility
        vib_flags_model: nn.Module = None,  # kept as is for compatibility
        # NOTE: you must carefully set the following parameters
        in_rest_idx=1,
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        lf0_teacher_forcing=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.lf0_teacher_forcing = lf0_teacher_forcing

        assert len(stream_sizes) in [4]

        self.encoder = encoder
        if self.encoder is not None:
            assert not encoder.is_autoregressive()
        self.mgc_model = mgc_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.bap_model = bap_model
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def has_residual_lf0_prediction(self):
        return True

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim

        if y is not None:
            # Teacher-forcing
            y_mgc, y_lf0, y_vuv, y_bap = split_streams(y, self.stream_sizes)
        else:
            # Inference
            y_mgc, y_lf0, y_vuv, y_bap = None, None, None, None

        # Predict continuous log-F0 first
        lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)

        if self.encoder is not None:
            encoder_outs = self.encoder(x, lengths)
            # Concat log-F0, rest flags and the outputs of the encoder
            # This may make the decoder to be aware of the input F0
            rest_flags = x[:, :, self.in_rest_idx].unsqueeze(-1)
            if self.lf0_teacher_forcing and y is not None:
                encoder_outs = torch.cat([encoder_outs, rest_flags, y_lf0], dim=-1)
            else:
                encoder_outs = torch.cat([encoder_outs, rest_flags, lf0], dim=-1)
        else:
            encoder_outs = x

        # Decoders for each stream
        mgc = self.mgc_model(encoder_outs, lengths, y_mgc)
        vuv = self.vuv_model(encoder_outs, lengths, y_vuv)
        bap = self.bap_model(encoder_outs, lengths, y_bap)

        # make a concatenated stream
        has_postnet_output = (
            isinstance(mgc, list)
            or isinstance(lf0, list)
            or isinstance(vuv, list)
            or isinstance(bap, list)
        )
        if has_postnet_output:
            outs = []
            for idx in range(len(mgc)):
                mgc_ = mgc[idx] if isinstance(mgc, list) else mgc
                lf0_ = lf0[idx] if isinstance(lf0, list) else lf0
                vuv_ = vuv[idx] if isinstance(vuv, list) else vuv
                bap_ = bap[idx] if isinstance(bap, list) else bap
                out = torch.cat([mgc_, lf0_, vuv_, bap_], dim=-1)
                assert out.shape[-1] == self.out_dim
                outs.append(out)
            return outs, lf0_residual
        else:
            out = torch.cat([mgc, lf0, vuv, bap], dim=-1)
            assert out.shape[-1] == self.out_dim

        return out, lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class MultiSpeakerMultistreamSeparateF0ParametricModel(BaseModel):
    """Multi-stream model with a separate F0 prediction model

    acoustic features: [MGC, LF0, VUV, BAP]

    vib_model and vib_flags_model are optional and will be likely to be removed.

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mgc_model (nn.Module): MGC prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        bap_model (nn.Module): BAP prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        lf0_teacher_forcing (bool): Whether to use teacher forcing for F0 prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        encoder: nn.Module,
        mgc_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        bap_model: nn.Module,
        speaker_embedding: nn.Module,
        vib_model: nn.Module = None,  # kept as is for compatibility
        vib_flags_model: nn.Module = None,  # kept as is for compatibility
        # NOTE: you must carefully set the following parameters
        in_rest_idx=1,
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        lf0_teacher_forcing=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.lf0_teacher_forcing = lf0_teacher_forcing

        assert len(stream_sizes) in [4]

        self.encoder = encoder
        if self.encoder is not None:
            assert not encoder.is_autoregressive()
        self.mgc_model = mgc_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.bap_model = bap_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def has_residual_lf0_prediction(self):
        return True

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def forward(self, x, spks, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim

        if y is not None:
            # Teacher-forcing
            y_mgc, y_lf0, y_vuv, y_bap = split_streams(y, self.stream_sizes)
        else:
            # Inference
            y_mgc, y_lf0, y_vuv, y_bap = None, None, None, None

        # concat speaker embedding
        spk_embs = self.speaker_embedding(spks)
        # spk_embs: (1, 256)
        spk_embs = spk_embs.expand(spk_embs.shape[0], x.shape[1], spk_embs.shape[-1])
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        # add speaker_embedding after adding phoneme embedding
        lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0, spk_embs)

        if self.encoder is not None:
            encoder_outs = self.encoder(x, lengths, spk_embs=spk_embs)
            # Concat log-F0, rest flags and the outputs of the encoder
            # This may make the decoder to be aware of the input F0
            rest_flags = x[:, :, self.in_rest_idx].unsqueeze(-1)
            if self.lf0_teacher_forcing and y is not None:
                encoder_outs = torch.cat([encoder_outs, rest_flags, y_lf0], dim=-1)
            else:
                encoder_outs = torch.cat([encoder_outs, rest_flags, lf0], dim=-1)
        else:
            encoder_outs = x

        # Decoders for each stream
        mgc = self.mgc_model(encoder_outs, lengths, y_mgc)
        vuv = self.vuv_model(encoder_outs, lengths, y_vuv)
        bap = self.bap_model(encoder_outs, lengths, y_bap)

        # make a concatenated stream
        has_postnet_output = (
            isinstance(mgc, list)
            or isinstance(lf0, list)
            or isinstance(vuv, list)
            or isinstance(bap, list)
        )
        if has_postnet_output:
            outs = []
            for idx in range(len(mgc)):
                mgc_ = mgc[idx] if isinstance(mgc, list) else mgc
                lf0_ = lf0[idx] if isinstance(lf0, list) else lf0
                vuv_ = vuv[idx] if isinstance(vuv, list) else vuv
                bap_ = bap[idx] if isinstance(bap, list) else bap
                out = torch.cat([mgc_, lf0_, vuv_, bap_], dim=-1)
                assert out.shape[-1] == self.out_dim
                outs.append(out)
            return outs, lf0_residual
        else:
            out = torch.cat([mgc, lf0, vuv, bap], dim=-1)
            assert out.shape[-1] == self.out_dim

        return out, lf0_residual

    def inference(self, x, spks=None, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            spks=spks,
        )


class MultiTrackMultistreamSeparateF0ParametricModel(BaseModel):
    """Multi-stream model with a separate F0 prediction model

    acoustic features: [MGC, LF0, VUV, BAP]

    vib_model and vib_flags_model are optional and will be likely to be removed.

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mgc_model (nn.Module): MGC prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        bap_model (nn.Module): BAP prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        lf0_teacher_forcing (bool): Whether to use teacher forcing for F0 prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        encoder: nn.Module,
        mgc_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        bap_model: nn.Module,
        speaker_embedding: nn.Module,
        vib_model: nn.Module = None,  # kept as is for compatibility
        vib_flags_model: nn.Module = None,  # kept as is for compatibility
        # NOTE: you must carefully set the following parameters
        in_rest_idx=1,
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        lf0_teacher_forcing=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.lf0_teacher_forcing = lf0_teacher_forcing

        assert len(stream_sizes) in [4]

        self.encoder = encoder
        if self.encoder is not None:
            assert not encoder.is_autoregressive()
        self.mgc_model = mgc_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.bap_model = bap_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def has_residual_lf0_prediction(self):
        return True

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def forward(self, x_main, x_sub, spks_list, lengths=None, ys=None):
        self._set_lf0_params()
        assert x_main.shape[-1] == self.in_dim
        is_inference = ys is None

        if ys is not None:
            # Teacher-forcing
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = split_streams(
                ys[0], self.stream_sizes
            )
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = split_streams(
                ys[1], self.stream_sizes
            )
        else:
            # Inference
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = None, None, None, None
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = None, None, None, None

        # concat speaker embedding
        spk_embs0 = self.speaker_embedding(spks_list[0])
        spk_embs1 = self.speaker_embedding(spks_list[1])
        # spk_embs: (1, 256)
        spk_embs0 = spk_embs0.expand(
            spk_embs0.shape[0], x_main.shape[1], spk_embs0.shape[-1]
        )
        spk_embs1 = spk_embs1.expand(
            spk_embs1.shape[0], x_sub.shape[1], spk_embs1.shape[-1]
        )
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        # add speaker_embedding after adding phoneme embedding
        lf0_main, lf0_residual_main = self.lf0_model(
            x_main, x_sub, spk_embs0, spk_embs1, lengths, y_lf0_main
        )
        lf0_sub, lf0_residual_sub = self.lf0_model(
            x_sub, x_main, spk_embs1, spk_embs0, lengths, y_lf0_sub
        )

        if self.encoder is not None:
            encoder_outs_main = self.encoder(
                x_main, x_sub, spk_embs=(spk_embs0, spk_embs1), lengths=lengths
            )
            encoder_outs_sub = self.encoder(
                x_sub, x_main, spk_embs=(spk_embs1, spk_embs0), lengths=lengths
            )
            # Concat log-F0, rest flags and the outputs of the encoder
            # This may make the decoder to be aware of the input F0
            rest_flags_main = x_main[:, :, self.in_rest_idx].unsqueeze(-1)
            rest_flags_sub = x_sub[:, :, self.in_rest_idx].unsqueeze(-1)
            if self.lf0_teacher_forcing and ys is not None:
                encoder_outs_main = torch.cat(
                    [encoder_outs_main, rest_flags_main, y_lf0_main], dim=-1
                )
                encoder_outs_sub = torch.cat(
                    [encoder_outs_sub, rest_flags_sub, y_lf0_sub], dim=-1
                )
            else:
                encoder_outs_main = torch.cat(
                    [encoder_outs_main, rest_flags_main, lf0_main], dim=-1
                )
                encoder_outs_sub = torch.cat(
                    [encoder_outs_sub, rest_flags_sub, lf0_sub], dim=-1
                )
        else:
            encoder_outs_main = x_main
            encoder_outs_sub = x_sub

        # Decoders for each stream
        mgc_main = self.mgc_model(encoder_outs_main, lengths, y_mgc_main)
        vuv_main = self.vuv_model(encoder_outs_main, lengths, y_vuv_main)
        bap_main = self.bap_model(encoder_outs_main, lengths, y_bap_main)
        mgc_sub = self.mgc_model(encoder_outs_main, lengths, y_mgc_sub)
        vuv_sub = self.vuv_model(encoder_outs_main, lengths, y_vuv_sub)
        bap_sub = self.bap_model(encoder_outs_main, lengths, y_bap_sub)

        # make a concatenated stream
        has_postnet_output_main = (
            isinstance(mgc_main, list)
            or isinstance(lf0_main, list)
            or isinstance(vuv_main, list)
            or isinstance(bap_main, list)
        )
        has_postnet_output_sub = (
            isinstance(mgc_sub, list)
            or isinstance(lf0_sub, list)
            or isinstance(vuv_sub, list)
            or isinstance(bap_sub, list)
        )
        if has_postnet_output_main and has_postnet_output_sub:
            outs_main = []
            for idx in range(len(mgc_main)):
                mgc_main_ = mgc_main[idx] if isinstance(mgc_main, list) else mgc_main
                lf0_main_ = lf0_main[idx] if isinstance(lf0_main, list) else lf0_main
                vuv_main_ = vuv_main[idx] if isinstance(vuv_main, list) else vuv_main
                bap_main_ = bap_main[idx] if isinstance(bap_main, list) else bap_main
                out_main = torch.cat(
                    [mgc_main_, lf0_main_, vuv_main_, bap_main_], dim=-1
                )
                assert out_main.shape[-1] == self.out_dim
                outs_main.append(out_sub)
            outs_sub = []
            for idx in range(len(mgc_sub)):
                mgc_sub_ = mgc_sub[idx] if isinstance(mgc_sub, list) else mgc_sub
                lf0_sub_ = lf0_sub[idx] if isinstance(lf0_sub, list) else lf0_sub
                vuv_sub_ = vuv_sub[idx] if isinstance(vuv_sub, list) else vuv_sub
                bap_sub_ = bap_sub[idx] if isinstance(bap_sub, list) else bap_sub
                out_sub = torch.cat([mgc_sub_, lf0_sub_, vuv_sub_, bap_sub_], dim=-1)
                assert out_sub.shape[-1] == self.out_dim
                outs_sub.append(out_sub)
            return (outs_main, lf0_residual_main), (outs_sub, lf0_residual_sub)
        else:
            out_main = torch.cat([mgc_main, lf0_main, vuv_main, bap_main], dim=-1)
            out_sub = torch.cat([mgc_sub, lf0_sub, vuv_sub, bap_sub], dim=-1)
            assert out_main.shape[-1] == self.out_dim
            assert out_sub.shape[-1] == self.out_dim

        if is_inference:
            return out_main, out_sub
        else:
            return (out_main, lf0_residual_main), (out_sub, lf0_residual_sub)

    def inference(self, x_main, x_sub, spks=None, lengths=None):
        return pad_inference_multitrack(
            model=self,
            x_main=x_main,
            x_sub=x_sub,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            spks=spks,
        )


class MultiTrackMultistreamSeparateF0ParametricModelv3(BaseModel):
    """Multi-stream model with a separate F0 prediction model

    acoustic features: [MGC, LF0, VUV, BAP]

    vib_model and vib_flags_model are optional and will be likely to be removed.

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mgc_model (nn.Module): MGC prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        bap_model (nn.Module): BAP prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        lf0_teacher_forcing (bool): Whether to use teacher forcing for F0 prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        encoder: nn.Module,
        mgc_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        bap_model: nn.Module,
        speaker_embedding: nn.Module,
        vib_model: nn.Module = None,  # kept as is for compatibility
        vib_flags_model: nn.Module = None,  # kept as is for compatibility
        # NOTE: you must carefully set the following parameters
        in_rest_idx=1,
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        lf0_teacher_forcing=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.lf0_teacher_forcing = lf0_teacher_forcing

        assert len(stream_sizes) in [4]

        self.encoder = encoder
        if self.encoder is not None:
            assert not encoder.is_autoregressive()
        self.mgc_model = mgc_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.bap_model = bap_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def has_residual_lf0_prediction(self):
        return True

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def forward(self, x_main, x_sub, spks_list, lengths=None, ys=None):
        self._set_lf0_params()
        assert x_main.shape[-1] == self.in_dim
        is_inference = ys is None

        if ys is not None:
            # Teacher-forcing
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = split_streams(
                ys[0], self.stream_sizes
            )
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = split_streams(
                ys[1], self.stream_sizes
            )
        else:
            # Inference
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = None, None, None, None
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = None, None, None, None

        # concat speaker embedding
        spk_embs0 = self.speaker_embedding(spks_list[0])
        spk_embs1 = self.speaker_embedding(spks_list[1])
        # spk_embs: (1, 256)
        spk_embs0 = spk_embs0.expand(
            spk_embs0.shape[0], x_main.shape[1], spk_embs0.shape[-1]
        )
        spk_embs1 = spk_embs1.expand(
            spk_embs1.shape[0], x_sub.shape[1], spk_embs1.shape[-1]
        )
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        # add speaker_embedding after adding phoneme embedding
        lf0_main, lf0_residual_main = self.lf0_model(
            x_main, x_sub, spk_embs0, spk_embs1, lengths, y_lf0_main
        )
        lf0_sub, lf0_residual_sub = self.lf0_model(
            x_sub, x_main, spk_embs1, spk_embs0, lengths, y_lf0_sub
        )

        if self.encoder is not None:
            encoder_outs_main = self.encoder(
                x_main, x_sub, spk_embs=(spk_embs0, spk_embs1), lengths=lengths
            )
            encoder_outs_sub = self.encoder(
                x_sub, x_main, spk_embs=(spk_embs1, spk_embs0), lengths=lengths
            )
            # Concat log-F0, rest flags and the outputs of the encoder
            # This may make the decoder to be aware of the input F0
            rest_flags_main = x_main[:, :, self.in_rest_idx].unsqueeze(-1)
            rest_flags_sub = x_sub[:, :, self.in_rest_idx].unsqueeze(-1)
            if self.lf0_teacher_forcing and ys is not None:
                encoder_outs_main = torch.cat(
                    [encoder_outs_main, rest_flags_main, y_lf0_main], dim=-1
                )
                encoder_outs_sub = torch.cat(
                    [encoder_outs_sub, rest_flags_sub, y_lf0_sub], dim=-1
                )
            else:
                encoder_outs_main = torch.cat(
                    [encoder_outs_main, rest_flags_main, lf0_main], dim=-1
                )
                encoder_outs_sub = torch.cat(
                    [encoder_outs_sub, rest_flags_sub, lf0_sub], dim=-1
                )
        else:
            encoder_outs_main = x_main
            encoder_outs_sub = x_sub
        encoder_outs = torch.cat([encoder_outs_main, encoder_outs_sub], dim=-1)

        # Decoders for each stream
        mgc_main = self.mgc_model(encoder_outs_main, lengths, y_mgc_main)
        vuv_main = self.vuv_model(encoder_outs_main, lengths, y_vuv_main)
        bap_main = self.bap_model(encoder_outs_main, lengths, y_bap_main)
        mgc_sub = self.mgc_model(encoder_outs_main, lengths, y_mgc_sub)
        vuv_sub = self.vuv_model(encoder_outs_main, lengths, y_vuv_sub)
        bap_sub = self.bap_model(encoder_outs_main, lengths, y_bap_sub)

        # make a concatenated stream
        has_postnet_output_main = (
            isinstance(mgc_main, list)
            or isinstance(lf0_main, list)
            or isinstance(vuv_main, list)
            or isinstance(bap_main, list)
        )
        has_postnet_output_sub = (
            isinstance(mgc_sub, list)
            or isinstance(lf0_sub, list)
            or isinstance(vuv_sub, list)
            or isinstance(bap_sub, list)
        )
        if has_postnet_output_main and has_postnet_output_sub:
            outs_main = []
            for idx in range(len(mgc_main)):
                mgc_main_ = mgc_main[idx] if isinstance(mgc_main, list) else mgc_main
                lf0_main_ = lf0_main[idx] if isinstance(lf0_main, list) else lf0_main
                vuv_main_ = vuv_main[idx] if isinstance(vuv_main, list) else vuv_main
                bap_main_ = bap_main[idx] if isinstance(bap_main, list) else bap_main
                out_main = torch.cat(
                    [mgc_main_, lf0_main_, vuv_main_, bap_main_], dim=-1
                )
                assert out_main.shape[-1] == self.out_dim
                outs_main.append(out_sub)
            outs_sub = []
            for idx in range(len(mgc_sub)):
                mgc_sub_ = mgc_sub[idx] if isinstance(mgc_sub, list) else mgc_sub
                lf0_sub_ = lf0_sub[idx] if isinstance(lf0_sub, list) else lf0_sub
                vuv_sub_ = vuv_sub[idx] if isinstance(vuv_sub, list) else vuv_sub
                bap_sub_ = bap_sub[idx] if isinstance(bap_sub, list) else bap_sub
                out_sub = torch.cat([mgc_sub_, lf0_sub_, vuv_sub_, bap_sub_], dim=-1)
                assert out_sub.shape[-1] == self.out_dim
                outs_sub.append(out_sub)
            return (outs_main, lf0_residual_main), (outs_sub, lf0_residual_sub)
        else:
            out_main = torch.cat([mgc_main, lf0_main, vuv_main, bap_main], dim=-1)
            out_sub = torch.cat([mgc_sub, lf0_sub, vuv_sub, bap_sub], dim=-1)
            assert out_main.shape[-1] == self.out_dim
            assert out_sub.shape[-1] == self.out_dim

        if is_inference:
            return out_main, out_sub
        else:
            return (out_main, lf0_residual_main), (out_sub, lf0_residual_sub)

    def inference(self, x_main, x_sub, spks=None, lengths=None):
        return pad_inference_multitrack(
            model=self,
            x_main=x_main,
            x_sub=x_sub,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            spks=spks,
        )


class NPSSMultistreamParametricModel(BaseModel):
    """NPSS-like cascaded multi-stream model with no mixture density networks.

    NPSS: :cite:t:`blaauw2017neural`

    Different from the original NPSS, we don't use spectral parameters
    for the inputs of aperiodicity and V/UV prediction models.
    This is because
    (1) D4C does not use spectral parameters as input for aperiodicity estimation.
    (2) V/UV detection is done from aperiodicity at 0-3 kHz in WORLD.
    In addition, f0 and VUV models dont use MDNs.

    Empirically, we found the above configuration works better than the original one.

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, BAP, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        lf0_model (BaseModel): Model for predicting log-F0.
        mgc_model (BaseModel): Model for predicting MGC.
        bap_model (BaseModel): Model for predicting BAP.
        vuv_model (BaseModel): Model for predicting V/UV.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_bap_conditioning (bool): If True, use  BAP features for V/UV prediction.
        vuv_model_bap0_conditioning (bool): If True, use only 0-th coef. of BAP
            for V/UV prediction.
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mgc_conditioning (bool): If True, use MGC features for V/UV prediction.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        lf0_model: nn.Module,
        mgc_model: nn.Module,
        bap_model: nn.Module,
        vuv_model: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        npss_style_conditioning=False,
        vuv_model_bap_conditioning=True,
        vuv_model_bap0_conditioning=False,
        vuv_model_lf0_conditioning=True,
        vuv_model_mgc_conditioning=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_bap_conditioning = vuv_model_bap_conditioning
        self.vuv_model_bap0_conditioning = vuv_model_bap0_conditioning
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mgc_conditioning = vuv_model_mgc_conditioning
        assert not npss_style_conditioning, "Not supported"

        assert len(stream_sizes) in [4]

        self.lf0_model = lf0_model
        self.mgc_model = mgc_model
        self.bap_model = bap_model
        self.vuv_model = vuv_model
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.DETERMINISTIC

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if is_inference:
            y_mgc, y_lf0, y_vuv, y_bap = (
                None,
                None,
                None,
                None,
            )
        else:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mgc, y_lf0, y_vuv, y_bap = outs

        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = self.lf0_model.inference(x, lengths), None
        else:
            lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)

        # Predict spectral parameters
        if is_inference:
            mgc_inp = torch.cat([x, lf0], dim=-1)
            mgc = self.mgc_model.inference(mgc_inp, lengths)
        else:
            mgc_inp = torch.cat([x, y_lf0], dim=-1)
            mgc = self.mgc_model(mgc_inp, lengths, y_mgc)

        # Predict aperiodic parameters
        if is_inference:
            bap_inp = torch.cat([x, lf0], dim=-1)
            bap = self.bap_model.inference(bap_inp, lengths)
        else:
            bap_inp = torch.cat([x, y_lf0], dim=-1)
            bap = self.bap_model(bap_inp, lengths, y_bap)

        # Predict V/UV
        if is_inference:
            if self.vuv_model_bap0_conditioning:
                bap_cond = bap[:, :, 0:1]
            else:
                bap_cond = bap
            # full cond: (x, mgc, lf0, bap)
            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(mgc)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(bap_cond)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(lf0)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            if self.vuv_model_bap0_conditioning:
                y_bap_cond = y_bap[:, :, 0:1]
            else:
                y_bap_cond = y_bap
            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(y_mgc)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(y_bap_cond)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(y_lf0)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        # make a concatenated stream
        has_postnet_output = (
            isinstance(mgc, list) or isinstance(bap, list) or isinstance(vuv, list)
        )
        if has_postnet_output:
            outs = []
            for idx in range(len(mgc)):
                mgc_ = mgc[idx] if isinstance(mgc, list) else mgc
                lf0_ = lf0[idx] if isinstance(lf0, list) else lf0
                vuv_ = vuv[idx] if isinstance(vuv, list) else vuv
                bap_ = bap[idx] if isinstance(bap, list) else bap
                out = torch.cat([mgc_, lf0_, vuv_, bap_], dim=-1)
                assert out.shape[-1] == self.out_dim
                outs.append(out)
        else:
            outs = torch.cat([mgc, lf0, vuv, bap], dim=-1)
            assert outs.shape[-1] == self.out_dim

        return outs, lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=False,
        )


class NPSSMDNMultistreamParametricModel(BaseModel):
    """NPSS-like cascaded multi-stream parametric model with mixture density networks.

    .. note::

        This class was originally designed to be used with MDNs. However, the internal
        design was changed to make it work with non-MDN and diffusion models. For example,
        you can use non-MDN models for MGC prediction.

    NPSS: :cite:t:`blaauw2017neural`

    acoustic features: [MGC, LF0, VUV, BAP]

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, BAP, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        lf0_model (BaseModel): Model for predicting log-F0.
        mgc_model (BaseModel): Model for predicting MGC.
        bap_model (BaseModel): Model for predicting BAP.
        vuv_model (BaseModel): Model for predicting V/UV.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_bap_conditioning (bool): If True, use  BAP features for V/UV prediction.
        vuv_model_bap0_conditioning (bool): If True, use only 0-th coef. of BAP
            for V/UV prediction.
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mgc_conditioning (bool): If True, use MGC features for V/UV prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        lf0_model: nn.Module,
        mgc_model: nn.Module,
        bap_model: nn.Module,
        vuv_model: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        vuv_model_bap_conditioning=True,
        vuv_model_bap0_conditioning=False,
        vuv_model_lf0_conditioning=True,
        vuv_model_mgc_conditioning=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_bap_conditioning = vuv_model_bap_conditioning
        self.vuv_model_bap0_conditioning = vuv_model_bap0_conditioning
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mgc_conditioning = vuv_model_mgc_conditioning

        assert len(stream_sizes) in [4]

        self.lf0_model = lf0_model
        self.mgc_model = mgc_model
        self.bap_model = bap_model
        self.vuv_model = vuv_model
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if is_inference:
            y_mgc, y_lf0, y_vuv, y_bap = (
                None,
                None,
                None,
                None,
            )
        else:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mgc, y_lf0, y_vuv, y_bap = outs

        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = self.lf0_model.inference(x, lengths), None
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond = lf0[0]
            else:
                lf0_cond = lf0
        else:
            lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)
        # Predict spectral parameters
        if is_inference:
            mgc_inp = torch.cat([x, lf0_cond], dim=-1)
            mgc = self.mgc_model.inference(mgc_inp, lengths)
        else:
            mgc_inp = torch.cat([x, y_lf0], dim=-1)
            mgc = self.mgc_model(mgc_inp, lengths, y_mgc)

        # Predict aperiodic parameters
        if is_inference:
            bap_inp = torch.cat([x, lf0_cond], dim=-1)
            bap = self.bap_model.inference(bap_inp, lengths)
        else:
            bap_inp = torch.cat([x, y_lf0], dim=-1)
            bap = self.bap_model(bap_inp, lengths, y_bap)

        # Predict V/UV
        if is_inference:
            if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                bap_cond = bap[0]
            else:
                bap_cond = bap
            if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                mgc_cond = mgc[0]
            else:
                mgc_cond = mgc

            if self.vuv_model_bap0_conditioning:
                bap_cond = bap_cond[:, :, 0:1]

            # full cond: (x, mgc, lf0, bap)
            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(mgc_cond)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(lf0_cond)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(bap_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            if self.vuv_model_bap0_conditioning:
                y_bap_cond = y_bap[:, :, 0:1]
            else:
                y_bap_cond = y_bap

            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(y_mgc)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(y_lf0)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(y_bap_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                bap_ = bap[0]
            else:
                bap_ = bap
            if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                mgc_ = mgc[0]
            else:
                mgc_ = mgc
            out = torch.cat([mgc_, lf0_, vuv, bap_], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mgc, lf0, vuv, bap), lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class MultiSpeakerNPSSMDNMultistreamParametricModel(BaseModel):
    """NPSS-like cascaded multi-stream parametric model with mixture density networks.

    .. note::

        This class was originally designed to be used with MDNs. However, the internal
        design was changed to make it work with non-MDN and diffusion models. For example,
        you can use non-MDN models for MGC prediction.

    NPSS: :cite:t:`blaauw2017neural`

    acoustic features: [MGC, LF0, VUV, BAP]

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, BAP, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        lf0_model (BaseModel): Model for predicting log-F0.
        mgc_model (BaseModel): Model for predicting MGC.
        bap_model (BaseModel): Model for predicting BAP.
        vuv_model (BaseModel): Model for predicting V/UV.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_bap_conditioning (bool): If True, use  BAP features for V/UV prediction.
        vuv_model_bap0_conditioning (bool): If True, use only 0-th coef. of BAP
            for V/UV prediction.
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mgc_conditioning (bool): If True, use MGC features for V/UV prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        lf0_model: nn.Module,
        mgc_model: nn.Module,
        bap_model: nn.Module,
        vuv_model: nn.Module,
        speaker_embedding: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        vuv_model_bap_conditioning=True,
        vuv_model_bap0_conditioning=False,
        vuv_model_lf0_conditioning=True,
        vuv_model_mgc_conditioning=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_bap_conditioning = vuv_model_bap_conditioning
        self.vuv_model_bap0_conditioning = vuv_model_bap0_conditioning
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mgc_conditioning = vuv_model_mgc_conditioning

        assert len(stream_sizes) in [4]

        self.lf0_model = lf0_model
        self.mgc_model = mgc_model
        self.bap_model = bap_model
        self.vuv_model = vuv_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, spks, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if is_inference:
            y_mgc, y_lf0, y_vuv, y_bap = (
                None,
                None,
                None,
                None,
            )
        else:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mgc, y_lf0, y_vuv, y_bap = outs

        # concat speaker embedding
        spk_embs = self.speaker_embedding(spks)
        # spk_embs: (1, 256)
        spk_embs = spk_embs.expand(spk_embs.shape[0], x.shape[1], spk_embs.shape[-1])
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = (
                self.lf0_model.inference(x, lengths=lengths, spk_embs=spk_embs),
                None,
            )
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond = lf0[0]
            else:
                lf0_cond = lf0
        else:
            lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0, spk_embs=spk_embs)

        # Predict spectral parameters
        if is_inference:
            mgc_inp = torch.cat([x, lf0_cond], dim=-1)
            mgc = self.mgc_model.inference(mgc_inp, lengths, spk_embs=spk_embs)
        else:
            mgc_inp = torch.cat([x, y_lf0], dim=-1)
            mgc = self.mgc_model(mgc_inp, lengths, y_mgc, spk_embs=spk_embs)
            if isinstance(mgc, list):
                mgc = mgc[1]  # use postfiltered one. Also see BiLSTMNonAttentiveDecoder

        # Predict aperiodic parameters
        if is_inference:
            bap_inp = torch.cat([x, lf0_cond], dim=-1)
            bap = self.bap_model.inference(bap_inp, lengths, spk_embs=spk_embs)
        else:
            bap_inp = torch.cat([x, y_lf0], dim=-1)
            bap = self.bap_model(bap_inp, lengths, y_bap, spk_embs=spk_embs)
            if isinstance(bap, list):
                bap = bap[1]  # use postfiltered one. Also see BiLSTMNonAttentiveDecoder

        # Predict V/UV
        if is_inference:
            if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                bap_cond = bap[0]
            else:
                bap_cond = bap
            if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                mgc_cond = mgc[0]
            else:
                mgc_cond = mgc

            if self.vuv_model_bap0_conditioning:
                bap_cond = bap_cond[:, :, 0:1]

            # full cond: (x, mgc, lf0, bap)
            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(mgc_cond)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(lf0_cond)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(bap_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model.inference(vuv_inp, lengths, spk_embs=spk_embs)
        else:
            if self.vuv_model_bap0_conditioning:
                y_bap_cond = y_bap[:, :, 0:1]
            else:
                y_bap_cond = y_bap

            vuv_inp = [x]
            if self.vuv_model_mgc_conditioning:
                vuv_inp.append(y_mgc)
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(y_lf0)
            if self.vuv_model_bap_conditioning:
                vuv_inp.append(y_bap_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv, spk_embs)

        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                bap_ = bap[0]
            else:
                bap_ = bap
            if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                mgc_ = mgc[0]
            else:
                mgc_ = mgc
            out = torch.cat([mgc_, lf0_, vuv, bap_], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mgc, lf0, vuv, bap), lf0_residual

    def inference(self, x, spks=None, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
            spks=spks,
        )


class MultiTrackNPSSMDNMultistreamParametricModel(BaseModel):
    """NPSS-like cascaded multi-stream parametric model with mixture density networks.

    .. note::

        This class was originally designed to be used with MDNs. However, the internal
        design was changed to make it work with non-MDN and diffusion models. For example,
        you can use non-MDN models for MGC prediction.

    NPSS: :cite:t:`blaauw2017neural`

    acoustic features: [MGC, LF0, VUV, BAP]

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, BAP, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        lf0_model (BaseModel): Model for predicting log-F0.
        mgc_model (BaseModel): Model for predicting MGC.
        bap_model (BaseModel): Model for predicting BAP.
        vuv_model (BaseModel): Model for predicting V/UV.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_bap_conditioning (bool): If True, use  BAP features for V/UV prediction.
        vuv_model_bap0_conditioning (bool): If True, use only 0-th coef. of BAP
            for V/UV prediction.
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mgc_conditioning (bool): If True, use MGC features for V/UV prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        lf0_model: nn.Module,
        mgc_model: nn.Module,
        bap_model: nn.Module,
        vuv_model: nn.Module,
        speaker_embedding: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        vuv_model_bap_conditioning=True,
        vuv_model_bap0_conditioning=False,
        vuv_model_lf0_conditioning=True,
        vuv_model_mgc_conditioning=False,
        output_subtrack=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_bap_conditioning = vuv_model_bap_conditioning
        self.vuv_model_bap0_conditioning = vuv_model_bap0_conditioning
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mgc_conditioning = vuv_model_mgc_conditioning
        self.output_subtrack = output_subtrack

        assert len(stream_sizes) in [4]

        self.lf0_model = lf0_model
        self.mgc_model = mgc_model
        self.bap_model = bap_model
        self.vuv_model = vuv_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x_main, x_sub, spks_list, lengths=None, ys=None):
        self._set_lf0_params()
        assert x_main.shape[-1] == self.in_dim
        is_inference = ys is None

        if is_inference:
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = (
                None,
                None,
                None,
                None,
            )
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = (
                None,
                None,
                None,
                None,
            )
        else:
            # Teacher-forcing
            outs_main = split_streams(ys[0], self.stream_sizes)
            outs_sub = split_streams(ys[1], self.stream_sizes)
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = outs_main
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = outs_sub

        # concat speaker embedding
        spk_embs0_orig = self.speaker_embedding(spks_list[0])
        spk_embs1_orig = self.speaker_embedding(spks_list[1])
        # spk_embs: (1, 256)
        spk_embs0 = spk_embs0_orig.expand(
            spk_embs0_orig.shape[0], x_main.shape[1], spk_embs0_orig.shape[-1]
        )
        spk_embs1 = spk_embs1_orig.expand(
            spk_embs1_orig.shape[0], x_sub.shape[1], spk_embs1_orig.shape[-1]
        )
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        if is_inference:
            lf0_main, lf0_residual_main = self.lf0_model(
                x_main, x_sub, spk_embs0, spk_embs1, lengths, y_lf0_main
            )
            lf0_sub, lf0_residual_sub = self.lf0_model(
                x_sub, x_main, spk_embs1, spk_embs0, lengths, y_lf0_sub
            )
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond_main = lf0_main[0]
                lf0_cond_sub = lf0_sub[0]
            else:
                lf0_cond_main = lf0_main
                lf0_cond_sub = lf0_sub
        else:
            lf0_main, lf0_residual_main = self.lf0_model(
                x_main, x_sub, spk_embs0, spk_embs1, lengths
            )
            lf0_sub, lf0_residual_sub = self.lf0_model(
                x_sub, x_main, spk_embs1, spk_embs0, lengths
            )
            lf0_cond_main = None
            lf0_cond_sub = None

        outs = []
        lf0_residuals = []
        for idx, (
            x,
            lf0,
            lf0_residual,
            lf0_cond,
            spk_emb,
            y_mgc,
            y_lf0,
            y_vuv,
            y_bap,
        ) in enumerate(
            zip(
                [x_main, x_sub],
                [lf0_main, lf0_sub],
                [lf0_residual_main, lf0_residual_sub],
                [lf0_cond_main, lf0_cond_sub],
                [spk_embs0, spk_embs1],
                [y_mgc_main, y_mgc_sub],
                [y_lf0_main, y_lf0_sub],
                [y_vuv_main, y_vuv_sub],
                [y_bap_main, y_bap_sub],
            )
        ):

            if idx == 0:
                # Predict spectral parameters
                if is_inference:
                    mgc_inp = torch.cat([x, lf0_cond], dim=-1)
                    mgc = self.mgc_model.inference(mgc_inp, lengths, spk_embs=spk_emb)
                else:
                    mgc_inp = torch.cat([x, y_lf0], dim=-1)
                    mgc = self.mgc_model(mgc_inp, lengths, y_mgc, spk_emb)

                # Predict aperiodic parameters
                if is_inference:
                    bap_inp = torch.cat([x, lf0_cond], dim=-1)
                    bap = self.bap_model.inference(bap_inp, lengths, spk_embs=spk_emb)
                else:
                    bap_inp = torch.cat([x, y_lf0], dim=-1)
                    bap = self.bap_model(bap_inp, lengths, y_bap, spk_emb)

                # Predict V/UV
                if is_inference:
                    if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                        bap_cond = bap[0]
                    else:
                        bap_cond = bap
                    if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                        mgc_cond = mgc[0]
                    else:
                        mgc_cond = mgc

                    if self.vuv_model_bap0_conditioning:
                        bap_cond = bap_cond[:, :, 0:1]

                    # full cond: (x, mgc, lf0, bap)
                    vuv_inp = [x]
                    if self.vuv_model_mgc_conditioning:
                        vuv_inp.append(mgc_cond)
                    if self.vuv_model_lf0_conditioning:
                        vuv_inp.append(lf0_cond)
                    if self.vuv_model_bap_conditioning:
                        vuv_inp.append(bap_cond)
                    vuv_inp = torch.cat(vuv_inp, dim=-1)
                    vuv = self.vuv_model.inference(vuv_inp, lengths, spk_embs=spk_emb)
                else:
                    if self.vuv_model_bap0_conditioning:
                        y_bap_cond = y_bap[:, :, 0:1]
                    else:
                        y_bap_cond = y_bap

                    vuv_inp = [x]
                    if self.vuv_model_mgc_conditioning:
                        vuv_inp.append(y_mgc)
                    if self.vuv_model_lf0_conditioning:
                        vuv_inp.append(y_lf0)
                    if self.vuv_model_bap_conditioning:
                        vuv_inp.append(y_bap_cond)
                    vuv_inp = torch.cat(vuv_inp, dim=-1)
                    vuv = self.vuv_model(vuv_inp, lengths, y_vuv, spk_emb)

            if is_inference:
                if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                    lf0_ = lf0[0]
                else:
                    lf0_ = lf0
                if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                    bap_ = bap[0]
                else:
                    bap_ = bap
                if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                    mgc_ = mgc[0]
                else:
                    mgc_ = mgc
                out = torch.cat([mgc_, lf0_, vuv, bap_], dim=-1)
                assert out.shape[-1] == self.out_dim

                return out, out
            else:
                if not self.output_subtrack:
                    return ((mgc, lf0, vuv, bap), lf0_residual), (None, None)

                if idx == 0:
                    outs.append((mgc, lf0, vuv, bap))
                else:
                    outs.append(
                        (y_mgc, lf0, y_vuv, y_bap)
                    )  # cut some features that are not used in interaction loss
                lf0_residuals.append(lf0_residual)

        assert not is_inference
        return (outs[0], lf0_residuals[0]), (outs[1], lf0_residuals[1])

    def inference(self, x_main, x_sub, spks=None, lengths=None):
        return pad_inference_multitrack(
            model=self,
            x_main=x_main,
            x_sub=x_sub,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            spks=spks,
        )


class V2MultiTrackNPSSMDNMultistreamParametricModel(BaseModel):
    """NPSS-like cascaded multi-stream parametric model with mixture density networks.

    .. note::

        This class was originally designed to be used with MDNs. However, the internal
        design was changed to make it work with non-MDN and diffusion models. For example,
        you can use non-MDN models for MGC prediction.

    NPSS: :cite:t:`blaauw2017neural`

    acoustic features: [MGC, LF0, VUV, BAP]

    Conditional dependency:
    p(MGC, LF0, VUV, BAP |C) = p(LF0|C) p(MGC|LF0, C) p(BAP|LF0, C) p(VUV|LF0, BAP, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        lf0_model (BaseModel): Model for predicting log-F0.
        mgc_model (BaseModel): Model for predicting MGC.
        bap_model (BaseModel): Model for predicting BAP.
        vuv_model (BaseModel): Model for predicting V/UV.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_bap_conditioning (bool): If True, use  BAP features for V/UV prediction.
        vuv_model_bap0_conditioning (bool): If True, use only 0-th coef. of BAP
            for V/UV prediction.
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mgc_conditioning (bool): If True, use MGC features for V/UV prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        lf0_model: nn.Module,
        mgc_model: nn.Module,
        bap_model: nn.Module,
        vuv_model: nn.Module,
        speaker_embedding: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        vuv_model_bap_conditioning=True,
        vuv_model_bap0_conditioning=False,
        vuv_model_lf0_conditioning=True,
        vuv_model_mgc_conditioning=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_bap_conditioning = vuv_model_bap_conditioning
        self.vuv_model_bap0_conditioning = vuv_model_bap0_conditioning
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mgc_conditioning = vuv_model_mgc_conditioning

        assert len(stream_sizes) in [4]

        self.lf0_model = lf0_model
        self.mgc_model = mgc_model
        self.bap_model = bap_model
        self.vuv_model = vuv_model
        self.speaker_embedding = speaker_embedding
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mgc_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
            or self.bap_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x_main, x_sub, spks_list, lengths=None, ys=None):
        self._set_lf0_params()
        assert x_main.shape[-1] == self.in_dim
        is_inference = ys is None
        if is_inference:
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = (
                None,
                None,
                None,
                None,
            )
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = (
                None,
                None,
                None,
                None,
            )
        else:
            # Teacher-forcing
            outs_main = split_streams(ys[0], self.stream_sizes)
            outs_sub = split_streams(ys[1], self.stream_sizes)
            y_mgc_main, y_lf0_main, y_vuv_main, y_bap_main = outs_main
            y_mgc_sub, y_lf0_sub, y_vuv_sub, y_bap_sub = outs_sub

        # concat speaker embedding
        spk_embs0 = self.speaker_embedding(spks_list[0])
        spk_embs1 = self.speaker_embedding(spks_list[1])
        # spk_embs: (1, 256)
        spk_embs0 = spk_embs0.expand(
            spk_embs0.shape[0], x_main.shape[1], spk_embs0.shape[-1]
        )
        spk_embs1 = spk_embs1.expand(
            spk_embs1.shape[0], x_sub.shape[1], spk_embs1.shape[-1]
        )
        # spk_embs: (1, T(num_frames), 256)

        # Predict continuous log-F0 first
        if is_inference:
            lf0_main, lf0_residual_main = self.lf0_model(
                x_main, x_sub, spk_embs0, spk_embs1, lengths, y_lf0_main
            )
            lf0_sub, lf0_residual_sub = self.lf0_model(
                x_sub, x_main, spk_embs1, spk_embs0, lengths, y_lf0_sub
            )
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond_main = lf0_main[0]
                lf0_cond_sub = lf0_sub[0]
            else:
                lf0_cond_main = lf0_main
                lf0_cond_sub = lf0_sub
        else:
            lf0_main, lf0_residual_main = self.lf0_model(
                x_main, x_sub, spk_embs0, spk_embs1, lengths
            )
            lf0_sub, lf0_residual_sub = self.lf0_model(
                x_sub, x_main, spk_embs1, spk_embs0, lengths
            )
            lf0_cond_main = None
            lf0_cond_sub = None

        outs = []
        lf0_residuals = []
        for x, lf0, lf0_residual, lf0_cond, spks, y_mgc, y_lf0, y_vuv, y_bap in zip(
            [x_main, x_sub],
            [lf0_main, lf0_sub],
            [lf0_residual_main, lf0_residual_sub],
            [lf0_cond_main, lf0_cond_sub],
            spks_list,
            [y_mgc_main, y_mgc_sub],
            [y_lf0_main, y_lf0_sub],
            [y_vuv_main, y_vuv_sub],
            [y_bap_main, y_bap_sub],
        ):
            # Predict spectral parameters
            if is_inference:
                mgc_inp = torch.cat([x, lf0_cond], dim=-1)
                mgc = self.mgc_model.inference(mgc_inp, spks=spks, lengths=lengths)
            else:
                mgc_inp = torch.cat([x, y_lf0], dim=-1)
                mgc = self.mgc_model(mgc_inp, spks=spks, lengths=lengths, y=y_mgc)

            # Predict aperiodic parameters
            if is_inference:
                bap_inp = torch.cat([x, lf0_cond], dim=-1)
                bap = self.bap_model.inference(bap_inp, spks=spks, lengths=lengths)
            else:
                bap_inp = torch.cat([x, y_lf0], dim=-1)
                bap = self.bap_model(bap_inp, spks=spks, lengths=lengths, y=y_bap)

            # Predict V/UV
            if is_inference:
                if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                    bap_cond = bap[0]
                else:
                    bap_cond = bap
                if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                    mgc_cond = mgc[0]
                else:
                    mgc_cond = mgc

                if self.vuv_model_bap0_conditioning:
                    bap_cond = bap_cond[:, :, 0:1]

                # full cond: (x, mgc, lf0, bap)
                vuv_inp = [x]
                if self.vuv_model_mgc_conditioning:
                    vuv_inp.append(mgc_cond)
                if self.vuv_model_lf0_conditioning:
                    vuv_inp.append(lf0_cond)
                if self.vuv_model_bap_conditioning:
                    vuv_inp.append(bap_cond)
                vuv_inp = torch.cat(vuv_inp, dim=-1)
                vuv = self.vuv_model.inference(vuv_inp, spks, lengths)
            else:
                if self.vuv_model_bap0_conditioning:
                    y_bap_cond = y_bap[:, :, 0:1]
                else:
                    y_bap_cond = y_bap

                vuv_inp = [x]
                if self.vuv_model_mgc_conditioning:
                    vuv_inp.append(y_mgc)
                if self.vuv_model_lf0_conditioning:
                    vuv_inp.append(y_lf0)
                if self.vuv_model_bap_conditioning:
                    vuv_inp.append(y_bap_cond)
                vuv_inp = torch.cat(vuv_inp, dim=-1)
                vuv = self.vuv_model(vuv_inp, spks, lengths, y_vuv)

            if is_inference:
                if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                    lf0_ = lf0[0]
                else:
                    lf0_ = lf0
                if self.bap_model.prediction_type() == PredictionType.PROBABILISTIC:
                    bap_ = bap[0]
                else:
                    bap_ = bap
                if self.mgc_model.prediction_type() == PredictionType.PROBABILISTIC:
                    mgc_ = mgc[0]
                else:
                    mgc_ = mgc
                out = torch.cat([mgc_, lf0_, vuv, bap_], dim=-1)
                assert out.shape[-1] == self.out_dim

                # outs.append((out,out))
                return out, out
            else:
                return ((mgc, lf0, vuv, bap), lf0_residual), (None, None)
                outs.append((mgc, lf0, vuv, bap))
                lf0_residuals.append(lf0_residual)
        # return (outs[0], lf0_residuals[0]), (outs[1], lf0_residuals[1])

    def inference(self, x_main, x_sub, spks=None, lengths=None):
        return pad_inference_multitrack(
            model=self,
            x_main=x_main,
            x_sub=x_sub,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            spks=spks,
        )


class MultistreamSeparateF0MelModel(BaseModel):
    """Multi-stream model with a separate F0 prediction model (mel-version)

    Conditional dependency:
    p(MEL, LF0, VUV|C) = p(LF0|C) p(MEL|LF0, C) p(VUV|LF0, C)

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mel_model (nn.Module): MEL prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        encoder: nn.Module,
        mel_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=1,
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor

        assert len(stream_sizes) == 3

        self.encoder = encoder
        if self.encoder is not None:
            assert not encoder.is_autoregressive()
        self.mel_model = mel_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return (
            self.mel_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim

        if y is not None:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mel, y_lf0, y_vuv = outs
        else:
            # Inference
            y_mel, y_lf0, y_vuv = (
                None,
                None,
                None,
            )

        # Predict continuous log-F0 first
        lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)

        if self.encoder is not None:
            encoder_outs = self.encoder(x, lengths)
            # Concat log-F0, rest flags and the outputs of the encoder
            # This may make the decoder to be aware of the input F0
            rest_flags = x[:, :, self.in_rest_idx].unsqueeze(-1)
            if y is not None:
                encoder_outs = torch.cat([encoder_outs, rest_flags, y_lf0], dim=-1)
            else:
                encoder_outs = torch.cat([encoder_outs, rest_flags, lf0], dim=-1)
        else:
            encoder_outs = x

        # Decoders for each stream
        mel = self.mel_model(encoder_outs, lengths, y_mel)
        vuv = self.vuv_model(encoder_outs, lengths, y_vuv)

        # make a concatenated stream
        has_postnet_output = (
            isinstance(mel, list) or isinstance(lf0, list) or isinstance(vuv, list)
        )
        if has_postnet_output:
            outs = []
            for idx in range(len(mel)):
                mel_ = mel[idx] if isinstance(mel, list) else mel
                lf0_ = lf0[idx] if isinstance(lf0, list) else lf0
                vuv_ = vuv[idx] if isinstance(vuv, list) else vuv
                out = torch.cat([mel_, lf0_, vuv_], dim=-1)
                assert out.shape[-1] == self.out_dim
                outs.append(out)
            return outs, lf0_residual
        else:
            out = torch.cat(
                [
                    mel,
                    lf0,
                    vuv,
                ],
                dim=-1,
            )
            assert out.shape[-1] == self.out_dim

        return out, lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class MDNMultistreamSeparateF0MelModel(BaseModel):
    """Multi-stream model with a separate F0 model (mel-version) and mDN

    V/UV prediction is performed given a mel-spectrogram.

    Conditional dependency:
    p(MEL, LF0, VUV|C) = p(LF0|C) p(MEL|LF0, C) p(VUV|LF0, MEL, C)

    .. note::

        This class was originally designed to be used with MDNs. However, the internal
        design was changed to make it work with non-MDN and diffusion models. For example,
        you can use non-MDN models for mel prediction.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        stream_sizes (list): List of stream sizes.
        reduction_factor (int): Reduction factor.
        encoder (nn.Module): A shared encoder.
        mel_model (nn.Module): MEL prediction model.
        lf0_model (nn.Module): log-F0 prediction model.
        vuv_model (nn.Module): V/UV prediction model.
        in_rest_idx (int): Index of the rest symbol in the input features.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        vuv_model_lf0_conditioning (bool): If True, use log-F0 features for V/UV prediction.
        vuv_model_mel_conditioning (bool): If True, use mel features for V/UV prediction.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        mel_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_rest_idx=0,
        in_lf0_idx=51,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        vuv_model_lf0_conditioning=True,
        vuv_model_mel_conditioning=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.vuv_model_lf0_conditioning = vuv_model_lf0_conditioning
        self.vuv_model_mel_conditioning = vuv_model_mel_conditioning

        assert len(stream_sizes) in [3]

        self.mel_model = mel_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.in_rest_idx = in_rest_idx
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mel_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if y is not None:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mel, y_lf0, y_vuv = outs
        else:
            # Inference
            y_mel, y_lf0, y_vuv = (
                None,
                None,
                None,
            )

        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = self.lf0_model.inference(x, lengths), None
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond = lf0[0]
            else:
                lf0_cond = lf0
        else:
            lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)

        # Predict mel
        if is_inference:
            mel_inp = torch.cat([x, lf0_cond], dim=-1)
            mel = self.mel_model.inference(mel_inp, lengths)
        else:
            mel_inp = torch.cat([x, y_lf0], dim=-1)
            mel = self.mel_model(mel_inp, lengths, y_mel)

        # Predict V/UV
        if is_inference:
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_cond = mel[0]
            else:
                mel_cond = mel

            # full cond: (x, lf0, mel)
            vuv_inp = [x]
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(lf0_cond)
            if self.vuv_model_mel_conditioning:
                vuv_inp.append(mel_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            vuv_inp = [x]
            if self.vuv_model_lf0_conditioning:
                vuv_inp.append(y_lf0)
            if self.vuv_model_mel_conditioning:
                vuv_inp.append(y_mel)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_ = mel[0]
            else:
                mel_ = mel
            out = torch.cat([mel_, lf0_, vuv], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mel, lf0, vuv), lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )
