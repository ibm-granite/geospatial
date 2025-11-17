# Copyright contributors to the Terratorch project
import logging
import warnings
from enum import Enum
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from terratorch.datasets import HLSBands
from terratorch.datasets.utils import generate_bands_intervals
from terratorch.models.backbones.prithvi_mae import PrithviMAE, PrithviViT
from terratorch.models.backbones.select_patch_embed_weights import (
    select_patch_embed_weights,
)
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class S3Bands(Enum):
    Oa01_reflectance = "Oa01_reflectance"
    Oa02_reflectance = "Oa02_reflectance"
    Oa03_reflectance = "Oa03_reflectance"
    Oa04_reflectance = "Oa04_reflectance"
    Oa05_reflectance = "Oa05_reflectance"
    Oa06_reflectance = "Oa06_reflectance"
    Oa07_reflectance = "Oa07_reflectance"
    Oa08_reflectance = "Oa08_reflectance"
    Oa09_reflectance = "Oa09_reflectance"
    Oa10_reflectance = "Oa10_reflectance"
    Oa11_reflectance = "Oa11_reflectance"
    Oa12_reflectance = "Oa12_reflectance"
    Oa16_reflectance = "Oa16_reflectance"
    Oa17_reflectance = "Oa17_reflectance"
    Oa18_reflectance = "Oa18_reflectance"
    Oa21_reflectance = "Oa21_reflectance"
    SST = "SST"

    @classmethod
    def try_convert_to_s3_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x


PRETRAINED_BANDS = [
    S3Bands.Oa01_reflectance,
    S3Bands.Oa02_reflectance,
    S3Bands.Oa03_reflectance,
    S3Bands.Oa04_reflectance,
    S3Bands.Oa05_reflectance,
    S3Bands.Oa06_reflectance,
    S3Bands.Oa07_reflectance,
    S3Bands.Oa08_reflectance,
    S3Bands.Oa09_reflectance,
    S3Bands.Oa10_reflectance,
    S3Bands.Oa11_reflectance,
    S3Bands.Oa12_reflectance,
    S3Bands.Oa16_reflectance,
    S3Bands.Oa17_reflectance,
    S3Bands.Oa18_reflectance,
    S3Bands.Oa21_reflectance,
    S3Bands.SST,
]
PRITHVI_S3_MEAN = [
    0.0235427398,
    0.0226303495,
    0.0199877248,
    0.0166938124,
    0.0119924026,
    0.00767917988,
    0.00251636861,
    0.00189688827,
    0.0019271833,
    0.0019056457,
    0.00103529217,
    0.00056689044,
    0.000595696267,
    0.000402757423,
    0.000423631744,
    0.000105166233,
    293.908469,
]
PRITHVI_S3_STD = [
    0.00776708,
    0.00733259,
    0.00633057,
    0.00615707,
    0.00610327,
    0.0066378,
    0.00539699,
    0.00511585,
    0.0050785,
    0.00507704,
    0.00484563,
    0.00415998,
    0.00441236,
    0.00408463,
    0.00400387,
    0.00370793,
    2.67577808,
]


def _cfg(**kwargs):
    return {
        "img_size": 42,
        "num_frames": 1,
        "patch_size": [1, 2, 2],
        "in_chans": 17,
        "embed_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "decoder_embed_dim": 256,
        "decoder_depth": 4,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
        "mean": PRITHVI_S3_MEAN,
        "std": PRITHVI_S3_STD,
        "coords_scale_learn": False,
        "bands": PRETRAINED_BANDS,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        **kwargs,
    }


prithvi_cfgs = {
    "prithvi_s3_v1": _cfg(
        num_frames=1,
        embed_dim=512,
        depth=12,
        num_heads=8,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        patch_size=[1, 2, 2],
        in_chans=17,
        coords_encoding=["day_length", "dtc", "region", "biome"],
    ),
}


# Timm pretrained configs
pretrained_weights = {
    "prithvi_s3_v1": {
        "hf_hub_id": "ibm-granite/granite-geospatial-ocean",
        "hf_hub_filename": "checkpoint.pt",
    }
}


def checkpoint_filter_fn_vit(
    state_dict,
    model: PrithviViT,
    pretrained_bands: list[S3Bands | int],
    model_bands: list[S3Bands | int],
) -> dict:
    """Encoder only model"""

    clean_dict = {}
    for k, v in state_dict.items():
        if "_timm_module." in k:  # Backwards compatibility for old model checkpoints
            k = k.replace("_timm_module.", "")

        if "pos_embed" in k:
            v = model.pos_embed  # pos_embed depends on num_frames and is fixed.
        if "decoder" in k or "_dec" in k or k == "mask_token":
            continue  # Drop decoder weights

        if k.startswith("encoder."):
            clean_dict[k.replace("encoder.", "")] = (
                v  # Convert Prithvi MAE to Prithvi ViT
            )
        else:
            clean_dict[k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(
        state_dict, model, pretrained_bands, model_bands
    )

    return state_dict


def checkpoint_filter_fn_mae(
    state_dict,
    model: PrithviMAE,
    pretrained_bands: list[S3Bands | int],
    model_bands: list[S3Bands | int],
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

        if k.startswith("encoder.") or k.startswith("decoder."):
            clean_dict[k] = v  # Weights in Prithvi MAE format
        # Convert Prithvi V1 weights
        elif "decoder" in k or "_dec" in k or k == "mask_token":
            clean_dict["decoder." + k] = v
        else:
            clean_dict["encoder." + k] = v

    state_dict = clean_dict

    state_dict = select_patch_embed_weights(
        state_dict, model, pretrained_bands, model_bands
    )

    return state_dict


def _create_prithvi(
    variant: str,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    model_bands: list[S3Bands | int] | None = None,
    ckpt_path: str = None,
    pretrained_bands: list[S3Bands | str | int] | None = None,
    num_frames: int = 1,
    encoder_only: bool = True,
    **kwargs,
) -> PrithviViT | PrithviMAE:
    """
    Build PrithviViT and PrithviMAE models.
    By default, encoder_only is set to True and a ViT is returned.
    """

    # Load default config
    model_args = prithvi_cfgs[variant].copy()

    # Backwards compatibility from timm (pretrained_cfg_overlay={"file": "<path to weights>"}) TODO: Remove before v1.0
    if "pretrained_cfg_overlay" in kwargs:
        warnings.warn(
            f"pretrained_cfg_overlay is deprecated and will be removed in a future version, "
            f"use ckpt_path=<file path> instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ckpt_path is not None:
            warnings.warn(
                f"pretrained_cfg_overlay and ckpt_path are provided, ignoring pretrained_cfg_overlay."
            )
        elif "file" not in kwargs["pretrained_cfg_overlay"]:
            warnings.warn(
                "pretrained_cfg_overlay does not include 'file path', ignoring pretrained_cfg_overlay."
            )
        else:
            ckpt_path = kwargs.pop("pretrained_cfg_overlay")["file"]

    pretrained_bands = pretrained_bands or model_args.get("bands", PRETRAINED_BANDS)

    if model_bands is None:
        model_bands: list[S3Bands | int] = pretrained_bands
        logger.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {pretrained_bands}."
            f"Pretrained patch_embed layer may be misaligned with current bands"
        )
    else:
        model_bands = [S3Bands.try_convert_to_s3_bands_enum(b) for b in model_bands]
        model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)
    kwargs["num_frames"] = num_frames
    model_args.update(kwargs)

    if encoder_only:
        prithvi_model_class = PrithviViT
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_vit
    else:
        prithvi_model_class = PrithviMAE
        checkpoint_filter_wrapper_fn = checkpoint_filter_fn_mae

    if pretrained:
        assert variant in pretrained_weights, (
            f"No pre-trained model found for variant {variant} "
            f"(pretrained models: {pretrained_weights.keys()})"
        )

    model = prithvi_model_class(**model_args)

    if ckpt_path is not None:
        # Load model from checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint_filter_wrapper_fn(
            state_dict, model, pretrained_bands, model_bands
        )
        model.load_state_dict(state_dict, strict=False)
    elif pretrained:
        try:
            # Download config.json to count model downloads
            _ = hf_hub_download(
                repo_id=pretrained_weights[variant]["hf_hub_id"], filename="config.json"
            )
            # Load model from Hugging Face
            pretrained_path = hf_hub_download(
                repo_id=pretrained_weights[variant]["hf_hub_id"],
                filename=pretrained_weights[variant]["hf_hub_filename"],
            )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint_filter_wrapper_fn(
                state_dict, model, pretrained_bands, model_bands
            )
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.error(f"Failed to load the pre-trained weights for {variant}.")
            raise e

    assert encoder_only or "out_indices" not in kwargs, (
        "out_indices provided for a MAE model."
    )
    if encoder_only:
        default_out_indices = list(range(len(model.blocks)))
        out_indices = kwargs.pop("out_indices", default_out_indices)

        def forward_filter_indices(*args, **kwargs):
            features = model.forward_features(*args, **kwargs)
            return [features[i] for i in out_indices]

        model.forward = forward_filter_indices
        model.out_indices = out_indices
        model.model_bands = model_bands
        model.pretrained_bands = pretrained_bands

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_s3_v1(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    bands: list[HLSBands] | None = None,
    **kwargs,
) -> PrithviViT:
    return _create_prithvi(
        "prithvi_s3_v1", pretrained=pretrained, model_bands=bands, **kwargs
    )
