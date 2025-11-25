# Copyright contributors to the Terratorch project
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from terratorch.datasets.utils import generate_bands_intervals
from terratorch.models.backbones.prithvi_mae import PrithviMAE, PrithviViT
from terratorch.models.backbones.prithvi_vit_adapter import PrithviViTAdapter
from terratorch.models.backbones.select_patch_embed_weights import (
    select_patch_embed_weights,
)
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from torch import nn

logger = logging.getLogger(__name__)


class S1HLSBands(Enum):
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"
    VV = "VV"
    VH = "VH"

    @classmethod
    def try_convert_to_hls_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x


PRETRAINED_BANDS = [
    S1HLSBands.BLUE,
    S1HLSBands.GREEN,
    S1HLSBands.RED,
    S1HLSBands.NIR_NARROW,
    S1HLSBands.SWIR_1,
    S1HLSBands.SWIR_2,
    S1HLSBands.VV,
    S1HLSBands.VH,
]

PRITHVI_V1_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
PRITHVI_V1_STD = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]
PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


def _cfg(**kwargs):
    return {
        "img_size": 224,
        "num_frames": 4,
        "patch_size": [1, 16, 16],
        "in_chans": 6,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
        "mean": PRITHVI_V2_MEAN,
        "std": PRITHVI_V2_STD,
        "coords_encoding": [],
        "coords_scale_learn": False,
        "bands": PRETRAINED_BANDS,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        "eps": 1e-6,
        **kwargs,
    }


prithvi_cfgs = {
    "granite_geospatial_uki": _cfg(
        num_frames=3, mean=PRITHVI_V1_MEAN, std=PRITHVI_V1_STD
    ),
}

pretrained_weights = {
    "granite_geospatial_uki": {
        "hf_hub_id": "ibm-granite/granite-geospatial-uki",
        "hf_hub_filename": "granite_geospatial_uki.pt",
    },
}

prithvi_adapter_cfgs = {
    "prithvi_eo_v1_100": {
        "interaction_indexes": [[0, 2], [3, 5], [6, 8], [9, 11]],
        "conv_inplane": 64,
        "add_vit_feature": True,
        "deform_num_heads": 12,
        "n_points": 4,
        "init_values": 0,
        "with_cffn": False,
        "cffn_ratio": 0.25,
        "deform_ratio": 0.5,
        "use_extra_extractor": True,
        "with_cp": False,
    },
    "prithvi_eo_v2_300": {
        "interaction_indexes": [[0, 5], [6, 11], [12, 17], [18, 23]],
        "conv_inplane": 64,
        "add_vit_feature": True,
        "deform_num_heads": 16,
        "n_points": 4,
        "init_values": 0,
        "with_cffn": False,
        "cffn_ratio": 0.25,
        "deform_ratio": 0.5,
        "use_extra_extractor": True,
        "with_cp": False,
    },
    "prithvi_eo_v2_300_tl": {
        "interaction_indexes": [[0, 5], [6, 11], [12, 17], [18, 23]],
        "conv_inplane": 64,
        "add_vit_feature": True,
        "deform_num_heads": 16,
        "n_points": 4,
        "init_values": 0,
        "with_cffn": False,
        "cffn_ratio": 0.25,
        "deform_ratio": 0.5,
        "use_extra_extractor": True,
        "with_cp": False,
    },
}


def checkpoint_filter_fn_vit(
    state_dict,
    model: PrithviViT,
    pretrained_bands: list[S1HLSBands | int],
    model_bands: list[S1HLSBands | int],
) -> dict:
    """Encoder only model"""

    clean_dict = {}
    for k, v in state_dict["state_dict"].items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

        # if "pos_embed" in k:
        #     v = model.pos_embed  # pos_embed depends on num_frames and is fixed.

        if "decoder" in k or "_dec" in k or k == "mask_token":
            continue  # Drop decoder weights

        if not model.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.location_encoding and "location_embed" in k:
            continue

        if k.startswith("encoder."):
            clean_dict[k.replace("encoder.", "")] = (
                v  # Convert Prithvi MAE to Prithvi ViT
            )
        else:
            clean_dict[k] = v

    for k, v in model.state_dict().items():
        if "vpt_prompt_embeddings" in k:
            clean_dict[k] = v

    state_dict["state_dict"] = clean_dict

    state_dict["state_dict"] = select_patch_embed_weights(
        state_dict, model, pretrained_bands, model_bands, encoder_only=True
    )

    return state_dict


def checkpoint_filter_fn_mae(
    state_dict,
    model: PrithviMAE,
    pretrained_bands: list[S1HLSBands | int],
    model_bands: list[S1HLSBands | int],
) -> dict:
    """Encoder-decoder model"""

    clean_dict = {}
    for k, v in state_dict.items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

        # pos_embed depends on num_frames and is fixed.
        if "decoder_pos_embed" in k:
            v = model.decoder.decoder_pos_embed
        elif "pos_embed" in k:
            v = model.encoder.pos_embed

        if not model.encoder.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.encoder.location_encoding and "location_embed" in k:
            continue

        if k.startswith("encoder.") or k.startswith("decoder."):
            clean_dict[k] = v  # Weights in Prithvi MAE format
        # Convert Prithvi V1 weights
        elif "decoder" in k or "_dec" in k or k == "mask_token":
            clean_dict["decoder." + k] = v
        else:
            clean_dict["encoder." + k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(
        state_dict,
        model,
        pretrained_bands,
        model_bands,
        encoder_only=False,
    )

    return state_dict


def checkpoint_filter_fn_vit_adapter(
    state_dict: dict,
    model: PrithviViTAdapter,
    pretrained_bands: list[S1HLSBands | str | int],
    model_bands: list[S1HLSBands | int],
) -> dict:
    state_dict = checkpoint_filter_fn_vit(
        state_dict, model, pretrained_bands, model_bands
    )
    clean_dict = {}
    for k, v in state_dict.items():
        if "cls_token" in k:
            continue
        if "norm." in k:
            continue
        clean_dict[k] = v

    for k, v in model.state_dict().items():
        if any(k.startswith(prefix) for prefix in PrithviViTAdapter.extra_layers):
            clean_dict[k] = v

    return clean_dict


def _create_prithvi(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    model_bands: list[S1HLSBands | int] | None = None,
    ckpt_path: str = None,
    pretrained_bands: list[S1HLSBands | str | int] | None = None,
    num_frames: int = 1,
    encoder_only: bool = True,
    vit_adapter: bool = False,
    **kwargs,
) -> PrithviViT | PrithviMAE | PrithviViTAdapter:
    """
    Build PrithviViT and PrithviMAE models.
    By default, encoder_only is set to True and a ViT is returned.
    """

    # Load default config
    model_args = prithvi_cfgs[variant].copy()

    if vit_adapter:
        if variant not in prithvi_adapter_cfgs:
            msg = (
                f"ViT Adapter not available for variant {variant}. "
                f"Available variants: {list(prithvi_adapter_cfgs.keys())}."
            )
            raise ValueError(msg)
        model_args |= prithvi_adapter_cfgs[variant].copy()

    # TODO: Remove in future version
    if "pretrained_cfg_overlay" in kwargs:
        raise ValueError(
            "pretrained_cfg_overlay was removed, use ckpt_path=<file path> instead."
        )

    pretrained_bands = pretrained_bands or model_args.get("bands", PRETRAINED_BANDS)

    if model_bands is None:
        model_bands: list[S1HLSBands | int] = pretrained_bands
        logger.info(
            f"model_bands not passed. Assuming bands are ordered in the same way as {pretrained_bands}."
            f"Pretrained patch_embed layer may be misaligned with current bands"
        )

    else:
        model_bands = [S1HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]
        model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)
    kwargs["num_frames"] = num_frames
    model_args.update(kwargs)

    prithvi_model_class: type[nn.Module]
    checkpoint_filter_wrapper_fn: Callable
    if vit_adapter:
        print("prithvi adapter")
        if not encoder_only:
            msg = "PrithviViTAdapter only supports encoder_only=True"
            raise ValueError(msg)
        prithvi_model_class = PrithviViTAdapter
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_vit_adapter
    elif encoder_only:
        print("PrithviViT model")
        prithvi_model_class = PrithviViT
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_vit
    else:
        print("PrithviMAE model")
        prithvi_model_class = PrithviMAE
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_mae

    model = prithvi_model_class(**model_args)

    if pretrained:
        if ckpt_path is not None:
            # Load model from checkpoint

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            if state_dict.get("state_dict"):
                state_dict = checkpoint_filter_wrapper_fn(
                    state_dict, model, pretrained_bands, model_bands
                )

            loaded_keys = model.load_state_dict(
                state_dict.get("state_dict", state_dict),
                strict=False,
            )

            if loaded_keys.missing_keys:
                logger.warning(
                    f"Missing keys in ckpt_path {ckpt_path}: {loaded_keys.missing_keys}"
                )
            if loaded_keys.unexpected_keys:
                logger.warning(
                    f"Unexpected keys in ckpt_path {ckpt_path}: {loaded_keys.unexpected_keys}"
                )

            if ("pos_embed" in loaded_keys.missing_keys) & (
                state_dict.get("state_dict") is not None
            ):

                class PossibleCheckpointVersionError(Exception):
                    def __init__(self, *args):
                        super().__init__(*args)

                message = (
                    "It looks like the checkpoint that is being used doesn't "
                    "contain positional embeddings or doesn't have the correct "
                    "naming convention. It's possible that you may be using an "
                    "older checkpoint from HuggingFace. We suggest "
                    "downloading the latest checkpoints at "
                    "https://huggingface.co/ibm-granite/granite-geospatial-uki-flooddetection/tree/main or "
                    "https://huggingface.co/ibm-granite/granite-geospatial-uki/tree/main. "
                )
                raise PossibleCheckpointVersionError(message)

        else:
            assert variant in pretrained_weights, (
                f"No pre-trained model found for variant {variant} (pretrained models: {pretrained_weights.keys()})"
            )

            try:
                # Load model from Hugging Face
                pretrained_path = hf_hub_download(
                    repo_id=pretrained_weights[variant]["hf_hub_id"],
                    filename=pretrained_weights[variant]["hf_hub_filename"],
                )
                state_dict = torch.load(
                    pretrained_path, map_location="cpu", weights_only=True
                )

                state_dict = checkpoint_filter_wrapper_fn(
                    state_dict, model, pretrained_bands, model_bands
                )
                model.load_state_dict(state_dict, strict=True)

            except RuntimeError as e:
                logger.error(f"Failed to load the pre-trained weights for {variant}.")
                raise e
    elif ckpt_path is not None:
        logger.warning(
            f"ckpt_path is provided but pretrained is set to False, ignoring ckpt_path {ckpt_path}."
        )

    # TODO Renanme to model.bands?
    model.model_bands = model_bands
    model.pretrained_bands = pretrained_bands

    assert encoder_only or "out_indices" not in kwargs, (
        "out_indices provided for a MAE model."
    )
    if vit_adapter:
        if "out_indices" in kwargs:
            msg = "out_indices should not be provided for ViTAdapter"
            raise ValueError(msg)
        model.forward = model.forward_features
        model.out_indices = [indexes[-1] for indexes in model.interaction_indexes]
    elif encoder_only:
        default_out_indices = list(range(len(model.blocks)))
        out_indices = kwargs.pop("out_indices", default_out_indices)

        def forward_filter_indices(*args, **kwargs):
            features = model.forward_features(*args, **kwargs)
            return [features[i] for i in out_indices]

        model.forward = forward_filter_indices
        model.out_indices = out_indices

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def granite_geospatial_uki(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[S1HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    model = _create_prithvi(
        "granite_geospatial_uki", pretrained=pretrained, model_bands=bands, **kwargs
    )

    # overwrite epsilon to match previous default value
    eps_value = 1e-6
    for n in range(len(model.blocks)):
        model.blocks[n].norm1.eps = eps_value
        model.blocks[n].norm2.eps = eps_value
    model.norm.eps = eps_value

    return model
