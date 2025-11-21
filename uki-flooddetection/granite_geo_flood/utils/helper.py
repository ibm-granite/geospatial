import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from global_land_mask import globe
from tifffile import imread
from torchmetrics.classification import (
    MulticlassFBetaScore,
    MulticlassJaccardIndex,
)
from xarray import DataArray


def overwrite_checkpoint_in_config(
    config_file: str | Path, new_checkpoint: str | Path
) -> None:
    """this will overwrite the 'backbone_ckpt_path' item in a terratorch config
    file with a checkpoint that a user specifies"""

    with open(config_file, "r") as f:
        config_contents = yaml.safe_load(f)

    config_contents["model"]["init_args"]["model_args"]["backbone_ckpt_path"] = str(
        new_checkpoint
    )

    with open(config_file, "w") as f:
        yaml.dump(config_contents, f, default_flow_style=False, sort_keys=False)


def download_data(region: str, save_file: str | Path) -> None:
    """script for downloading datasets prepared specifically for this repo.
    Includes UK and Ireland data and some images from Spain.

    Args:
        region (str): region name. Options are "uki", "uki_and_spain", "valencia".
        save_file (str | Path): file location and name with which you plan to save the downloaded data.
    """
    import sys

    # define URL locations
    match region:
        case "uki":
            gdrive_url = (
                "https://drive.google.com/uc?id=1pYQBorXCykfJpY-Bc_ce40yogMDJxmUJ"
            )
            zenodo_url = "https://zenodo.org/records/14216851/files/granite-geospatial-uki-flooddetection-dataset-uki.tar.gz?download=1"
        case "uki_and_spain":
            gdrive_url = (
                "https://drive.google.com/uc?id=1ysyyvLEQ8C05kmxTv0w91eOFHy2Z5V2i"
            )
            zenodo_url = "https://zenodo.org/records/14216851/files/granite-geospatial-uki-flooddetection-dataset-combined-uki-spain.tar.gz?download=1"
        case "valencia":
            gdrive_url = (
                "https://drive.google.com/uc?id=1CE1TV7WgpMi-jIuKmnvkbgZ_GYQUXPpk"
            )
            zenodo_url = "https://zenodo.org/records/14216851/files/granite-geospatial-uki-flooddetection-dataset-valencia.tar.gz?download=1"

    # downloading
    zenodo_command = f'wget "{zenodo_url}" -O {str(save_file)}'
    if "google.colab" in sys.modules:
        import gdown

        try:
            gdown.download(gdrive_url, str(save_file))
        except:
            print("Download failed via g.down. Reverting to Zenodo.")
            os.system(zenodo_command)
    else:
        os.system(zenodo_command)


def plot_images_pred_valencia(
    s1_before_flood: DataArray,
    s2_before_flood: DataArray,
    pred_before_flood: DataArray,
    s1_after_flood: DataArray,
    s2_after_flood: DataArray,
    flood_date_before: str,
    flood_date_after: str,
    pred_after_flood: DataArray,
    save_file: Path | str,
) -> None:
    """plot before and after flood events of Valencia region

    Args:
        s1_before_flood (DataArray): S1 VV before flood
        s2_before_flood (DataArray): S2 RGB before flood
        pred_before_flood (DataArray): inference results before flood
        s1_after_flood (DataArray): S1 VV after flood
        s2_after_flood (DataArray): S2 RGB after flood
        flood_date_before (str): date of imagery before flood
        flood_date_after (str): date of imagery after flood
        pred_after_flood (DataArray): inference results after flood
        save_file (Path | str): figure name for saving the plot
    """

    # set colorschemes for flood maps
    flood_cmap = mpl.colors.ListedColormap(["tan", "paleturquoise"])
    bounds = [-0.5, 0.5, 1.5]
    tick_vals = [0, 1]
    tick_labels = ["not water", "water"]
    norm = mpl.colors.BoundaryNorm(bounds, flood_cmap.N)

    # make the figure
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")

    ## before flood

    # S1
    axs[0, 0].imshow(s1_before_flood, cmap="afmhot")
    axs[0, 0].axis("off")
    axs[0, 0].set_title(f"{flood_date_before}: S1 - VV")

    # S2
    axs[0, 1].imshow(s2_before_flood)
    axs[0, 1].axis("off")
    axs[0, 1].set_title(f"{flood_date_before}: S2 - RGB")

    # pred map
    axs[0, 2].imshow(pred_before_flood, cmap=flood_cmap)
    axs[0, 2].axis("off")
    axs[0, 2].set_title(f"{flood_date_before}: predicted map")

    ## after flood

    # S1
    axs[1, 0].imshow(s1_after_flood, cmap="afmhot")
    axs[1, 0].axis("off")
    axs[1, 0].set_title(f"{flood_date_after}: S1 - VV")

    # S2
    axs[1, 1].imshow(s2_after_flood)
    axs[1, 1].axis("off")
    axs[1, 1].set_title(f"{flood_date_after}: S2 - RGB")

    # pred map
    axs[1, 2].imshow(pred_after_flood, cmap=flood_cmap)
    axs[1, 2].axis("off")
    axs[1, 2].set_title(f"{flood_date_after}: predicted map")

    # adjust colorbar
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=flood_cmap),
        ax=axs[:, -1],
        orientation="vertical",
        shrink=0.3,
    )
    cbar.set_ticks(ticks=tick_vals, labels=tick_labels)

    # save the figure
    plt.savefig(save_file)


def compare_images_label_pred(
    image_file: Path | str,
    label_file: Path | str,
    inference_file_mod1: Path | str,
    inference_file_mod2: Path | str,
    mod1_name: str,
    mod2_name: str,
    s1_band_id: int,
    s2_rgb_ids: list,
    save_dir: Path | str,
) -> None:
    """plots the S1 and S2 bands of the test set input images,
    and compares flood map truth labels with predicted flood maps.

    Args:
        image_file (str): tif file containing the S1 and S2 band images [h x w x bands]
        label_file (str): flood map tif file (truth labels)
        inference_file_mod1 (Path | str): flood map tif file of first model (inference result)
        inference_file_mod2 (Path | str): flood map tif file of second model (inference result)
        mod1_name (str): name of first model to annotate in figure
        mod2_name (str): name of second model to annotate in figure
        s1_band_id (int): VV band index number of the input image
        s2_rgb_ids (list): RGB bands of the input image
        save_dir (str): directory in which to save the figure
    """

    # load files
    input_image = imread(image_file)
    input_vv = input_image[:, :, s1_band_id]
    input_s2 = input_image[:, :, s2_rgb_ids]
    truth = imread(label_file)
    pred_mod1 = imread(inference_file_mod1)
    pred_mod2 = imread(inference_file_mod2)

    # check number of categories in the labels
    num_categories = np.unique(truth).shape[0]

    # make the figure
    fig, axs = plt.subplots(1, 5, figsize=(15, 4), layout="constrained")

    # S1
    plot_id = 0
    axs[plot_id].imshow(input_vv, cmap="afmhot")
    axs[plot_id].axis("off")
    axs[plot_id].set_title("S1 - VV")

    # S2
    input_s2 = scale_s2_image(input_s2)
    plot_id = 1
    axs[plot_id].imshow(input_s2)
    axs[plot_id].axis("off")
    axs[plot_id].set_title("S2 - RGB")

    # set colorschemes for flood maps
    match num_categories:
        case 2:
            flood_cmap = mpl.colors.ListedColormap(["tan", "paleturquoise"])
            bounds = [-0.5, 0.5, 1.5]
            tick_vals = [0, 1]
            tick_labels = ["not water", "water"]
        case 3:
            flood_cmap = mpl.colors.ListedColormap(["black", "tan", "paleturquoise"])
            bounds = [-1.5, -0.5, 0.5, 1.5]
            tick_vals = [-1, 0, 1]
            tick_labels = ["no data", "not water", "water"]

    norm = mpl.colors.BoundaryNorm(bounds, flood_cmap.N)

    # truth map
    plot_id = 2
    axs[plot_id].imshow(truth, cmap=flood_cmap)
    axs[plot_id].axis("off")
    axs[plot_id].set_title("truth map")

    # pred map
    plot_id = 3
    axs[plot_id].imshow(pred_mod1, cmap=flood_cmap)
    axs[plot_id].axis("off")
    axs[plot_id].set_title(f"predicted map: {mod1_name}")

    # pred map
    plot_id = 4
    axs[plot_id].imshow(pred_mod2, cmap=flood_cmap)
    axs[plot_id].axis("off")
    axs[plot_id].set_title(f"predicted map: {mod2_name}")

    # adjust colorbar
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=flood_cmap),
        ax=axs[:],
        orientation="horizontal",
        shrink=0.4,
    )
    cbar.set_ticks(ticks=tick_vals, labels=tick_labels)

    # save the figure
    os.makedirs(save_dir, exist_ok=True)
    filename = label_file.name.replace("_label.tif", "_inference_results.png")
    figure_name = save_dir / filename
    plt.savefig(figure_name)


def mask_image(image: DataArray) -> DataArray:
    """masking over oceans"""

    # get land mask
    lon_grid, lat_grid = np.meshgrid(image.x, image.y)
    globe_land_mask = globe.is_land(lat_grid, lon_grid)

    # mask land
    masked_image = image.where(cond=globe_land_mask, other=1)

    return masked_image


def clip_image(image: DataArray) -> DataArray:
    """clipping images for the Valencia region to work on free colab account"""
    image = image.rio.clip_box(
        minx=-0.3149,
        miny=39.1032,
        maxx=-0.2335,
        maxy=39.1701,
    )
    return image


def prep_valencia_images(
    image: DataArray,
    pred: DataArray,
    rgb_bands: list,
    vv_band: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """helps prepare and extract valencia images for plotting

    Args:
        image (DataArray): input image
        pred (DataArray): prediction from inference from input image
        rgb_bands (list): indices of RGB bands (in order) in image
        vv_band (int): index of S1 VV band in image

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: S1 vv band, S2 RGB, predictions
    """

    # apply masking over ocean as we don't train on it
    image = mask_image(image)
    pred = mask_image(pred)
    pred = pred.squeeze()

    # get S2 RGB bands
    s2 = image[rgb_bands, :, :]

    # move bands to the end to help with plotting
    s2 = s2.transpose("y", "x", "band")

    # get S1 - VV band
    s1 = image[vv_band, :, :]
    s1 = s1.squeeze()

    # scale S2 image for easy visualisation
    s2 = s2.to_numpy()
    s2 = scale_s2_image(s2)

    return s1.to_numpy(), s2, pred.to_numpy()


def scale_s2_image(image: np.ndarray) -> np.ndarray:
    """brighten darker RGB images for easy visualisation"""
    image[image < 0] = 0
    pl, ph = np.percentile(image, [2, 98])
    image = image / ph

    return image


def calc_metrics(truth_files: list, pred_files: list) -> dict:
    """calculating some simple metrics for model performance evaluation

    Args:
        truth_files (list): truth labels for a particular model
        pred_files (list): predicted labels for the same model

    Returns:
        dict: contains mIoU and F1 score
    """
    # load data into 3D array of size num_files x 512 x 512
    truth = np.array([imread(truth_file) for truth_file in truth_files])
    pred = np.array([imread(truth_file) for truth_file in pred_files])

    # convert from deta array to tensor
    truth = torch.tensor(truth)
    pred = torch.tensor(pred)

    # calculate mIoU
    miou = calc_miou(truth, pred)

    # calculate F1
    f1 = calc_f1(truth, pred)

    return {"mIoU": miou.detach().item(), "F1": f1.detach().item()}


def calc_miou(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """calculating mIoU"""
    metric = MulticlassJaccardIndex(num_classes=3, ignore_index=-1)
    return metric(truth, pred)


def calc_f1(truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """calculating f1 score"""
    metric = MulticlassFBetaScore(
        num_classes=3, ignore_index=-1, beta=1.0, average="micro"
    )
    return metric(truth, pred)


def gather_truth_and_pred(
    inf_dir: Path | str, label_dir: Path | str, search_dataset: str
) -> tuple[list, list]:
    """gathering truth labels and predected labels

    Args:
        inf_dir (Path | str): directory where predictions are kept
        label_dir (Path | str): directory where truth labels are kept
        search_dataset (str): any specific pattern you want to search for in
            the above two directories. e.g. "test" if you just want the test
            set only

    Returns:
        tuple[list, list]: ordered predicted label files, ordered truth label files
    """
    inf_dir = Path(inf_dir)
    pred_files = sorted(list(inf_dir.glob(f"E*{search_dataset}*pred.tif")))
    truth_files = sorted(list(label_dir.glob(f"E*{search_dataset}*label.tif")))

    return pred_files, truth_files
