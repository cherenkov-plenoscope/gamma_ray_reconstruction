import numpy as np
import binning_utils
from . import ellipse_model
from ... import utils


__fov_radius_deg__ = 3.25

EXAMPLE = {
    "image": {
        "radius_deg": __fov_radius_deg__ + 1.0,
        "num_bins": 128,
        "smoothing_kernel_width_deg": 0.3125,
    },
    "azimuth_ring": {
        "num_bins": 360,
        "radius_deg": 1.0,
        "smoothing_kernel_width_deg": 41.0,
    },
    "ellipse_model": ellipse_model.EXAMPLE_CONFIG,
}


def compile_user_config(user_config):
    uc = user_config
    cfg = {}

    uimg = uc["image"]
    img = {}
    img["radius"] = np.deg2rad(uimg["radius_deg"])
    img["num_bins"] = uimg["num_bins"]
    img["c_bin_edges"] = np.linspace(
        -img["radius"], +img["radius"], img["num_bins"] + 1,
    )
    img["c_bin_centers"] = binning_utils.centers(bin_edges=img["c_bin_edges"])
    _image_bins_per_rad = img["num_bins"] / (2.0 * img["radius"])
    img["smoothing_kernel_width"] = np.deg2rad(
        uimg["smoothing_kernel_width_deg"]
    )
    img["smoothing_kernel"] = utils.discrete_kernel.gauss2d(
        num_steps=int(
            np.round(img["smoothing_kernel_width"] * _image_bins_per_rad)
        )
    )
    cfg["image"] = img

    uazr = uc["azimuth_ring"]
    azr = {}
    azr["num_bins"] = uazr["num_bins"]
    azr["bin_edges"] = np.linspace(
        0.0, 2.0 * np.pi, azr["num_bins"], endpoint=False
    )
    azr["radius"] = np.deg2rad(uazr["radius_deg"])
    _ring_bins_per_rad = azr["num_bins"] / (2.0 * np.pi)
    azr["smoothing_kernel_width"] = np.deg2rad(
        uazr["smoothing_kernel_width_deg"]
    )
    azr["smoothing_kernel"] = utils.discrete_kernel.gauss1d(
        num_steps=int(
            np.round(_ring_bins_per_rad * azr["smoothing_kernel_width"])
        )
    )
    cfg["azimuth_ring"] = azr
    cfg["ellipse_model"] = dict(uc["ellipse_model"])
    return cfg
