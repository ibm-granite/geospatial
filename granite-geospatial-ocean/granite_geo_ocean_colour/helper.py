import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from skimage import exposure


def get_rgb(img):
    bands = [
        "Oa01_reflectance",
        "Oa02_reflectance",
        "Oa03_reflectance",
        "Oa04_reflectance",
        "Oa05_reflectance",
        "Oa06_reflectance",
        "Oa07_reflectance",
        "Oa08_reflectance",
        "Oa09_reflectance",
        "Oa10_reflectance",
        "Oa11_reflectance",
        "Oa12_reflectance",
        "Oa16_reflectance",
        "Oa17_reflectance",
        "Oa18_reflectance",
        "Oa21_reflectance",
    ]
    badpts = img[bands.index("Oa01_reflectance"), :, :] == -9999
    img = img / 63535

    # Natural colour broad band, log scaled
    red_recipe = (
        0.16666 * img[bands.index("Oa08_reflectance"), :, :]
        + 0.66666 * img[bands.index("Oa09_reflectance"), :, :]
        + 0.08333 * img[bands.index("Oa10_reflectance"), :, :]
        + 0.08333 * img[bands.index("Oa11_reflectance"), :, :]
    )
    green_recipe = (
        0.16666 * img[bands.index("Oa05_reflectance"), :, :]
        + 0.66666 * img[bands.index("Oa06_reflectance"), :, :]
        + 0.16666 * img[bands.index("Oa07_reflectance"), :, :]
    )
    blue_recipe = (
        0.16666 * img[bands.index("Oa02_reflectance"), :, :]
        + 0.66666 * img[bands.index("Oa03_reflectance"), :, :]
        + 0.16666 * img[bands.index("Oa04_reflectance"), :, :]
    )
    red_recipe.values[badpts.values] = 0
    green_recipe.values[badpts.values] = 0
    blue_recipe.values[badpts.values] = 0
    rgb = np.vstack(
        [
            np.expand_dims(red_recipe, axis=0),
            np.expand_dims(green_recipe, axis=0),
            np.expand_dims(blue_recipe, axis=0),
        ]
    )

    rgbb = rgb[0, :, :]
    rgbg = rgb[1, :, :]
    rgbr = rgb[2, :, :]

    plr, phr = np.percentile(rgbb[rgbb > 0], (1, 99))
    plg, phg = np.percentile(rgbg[rgbg > 0], (1, 99))
    plb, phb = np.percentile(rgbr[rgbb > 0], (1, 99))

    rgb_rescal = rgb.copy()
    rgb_rescal[0, :, :] = exposure.rescale_intensity(
        np.array(rgb[0, :, :]), in_range=(plr, phr)
    )
    rgb_rescal[1, :, :] = exposure.rescale_intensity(
        np.array(rgb[1, :, :]), in_range=(plg, phg)
    )
    rgb_rescal[2, :, :] = exposure.rescale_intensity(
        np.array(rgb[2, :, :]), in_range=(plb, phb)
    )
    rgb_rescal = rgb_rescal

    rgb_rescal = rgb_rescal.transpose([1, 2, 0])
    rgb_rescal[rgb_rescal > 1] = 1
    rgb_rescal[rgb_rescal < 0] = 0

    return rgb_rescal


def plot_training_data(selected_images):
    # make the figure
    fig, axs = plt.subplots(
        1, len(selected_images), figsize=(10, 6), layout="constrained"
    )

    for i, selected_image in enumerate(selected_images):
        selected_label = f"{selected_image.split('_img.tif')[0]}_lab.tif"
        img = rioxarray.open_rasterio(selected_image)
        lab = np.squeeze(rioxarray.open_rasterio(selected_label))

        # convert sentinel-3 olci image to natural colour image
        selected_image_rgb = get_rgb(img)
        lab_masked = np.ma.masked_where(lab < 0, lab)
        axs[i].imshow(selected_image_rgb)
        axs[i].axis("off")
        axs[i].imshow(lab_masked)


def crop_image(imput_image, bbox, output_image):
    img = rioxarray.open_rasterio(imput_image)
    geometries = [
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]],
                    [bbox[0], bbox[1]],
                ]
            ],
        }
    ]
    img_clip = img.rio.clip(geometries)
    img_clip.rio.to_raster(output_image)


def plot_inference_data(imput_image, inference_image):
    img = rioxarray.open_rasterio(imput_image)
    img_pred = rioxarray.open_rasterio(inference_image)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")

    # convert sentinel-3 olci image to natural colour image
    image_rgb = get_rgb(img)
    axs[0].imshow(image_rgb)
    axs[0].axis("off")

    img_pred_masked = np.ma.masked_where(np.squeeze(img_pred) < 0, np.squeeze(img_pred))
    axs[1].imshow(img_pred_masked, cmap="winter")
    axs[1].axis("off")
