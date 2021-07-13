from . import config
from . import ellipse_model
from . import ring_model
from . import fuzzy_model

import scipy.signal
import numpy as np


def estimate_main_axis_to_core(
    split_light_field,
    model_config,
    image_binning,
    image_smoothing_kernel,
    ring_binning,
    ring_smoothing_kernel,
):
    median_cx = split_light_field["median_cx"]
    median_cy = split_light_field["median_cy"]

    # make fuzzy image
    # ----------------
    # A probability-density for the shower's main-axis and primary particle's
    # direction.

    slf_model = ellipse_model.estimate_model_from_light_field(
        split_light_field=split_light_field, model_config=model_config
    )
    fuzzy_image = fuzzy_model.make_image_from_model(
        split_light_field_model=slf_model, image_binning=image_binning,
    )
    fuzzy_image_smooth = scipy.signal.convolve2d(
        in1=fuzzy_image, in2=image_smoothing_kernel, mode="same"
    )
    reco_cx, reco_cy = fuzzy_model.argmax_image_cx_cy(
        image=fuzzy_image_smooth, image_binning=image_binning,
    )

    median_cx_std = np.std([a["median_cx"] for a in slf_model])
    median_cy_std = np.std([a["median_cy"] for a in slf_model])

    # make ring to find main-axis
    # ---------------------------

    azimuth_ring = ring_model.project_image_onto_ring(
        image=fuzzy_image_smooth,
        image_binning=image_binning,
        ring_cx=median_cx,
        ring_cy=median_cy,
        ring_radius=ring_binning["radius"],
        ring_binning=ring_binning,
    )
    azimuth_ring_smooth = ring_model.circular_convolve1d(
        in1=azimuth_ring, in2=ring_smoothing_kernel
    )
    azimuth_ring_smooth /= np.max(azimuth_ring_smooth)

    # analyse ring to find main-axis
    # ------------------------------

    # maximum
    main_axis_azimuth = ring_binning["bin_edges"][
        np.argmax(azimuth_ring_smooth)
    ]

    # relative uncertainty
    _unc = np.mean(azimuth_ring_smooth)
    main_axis_azimuth_uncertainty = _unc ** 2.0

    result = {}
    result["main_axis_support_cx"] = median_cx
    result["main_axis_support_cy"] = median_cy
    result["main_axis_support_uncertainty"] = np.hypot(
        median_cx_std, median_cy_std
    )
    result["main_axis_azimuth"] = float(main_axis_azimuth)
    result["main_axis_azimuth_uncertainty"] = main_axis_azimuth_uncertainty
    result["reco_cx"] = reco_cx
    result["reco_cy"] = reco_cy

    debug = {}
    debug["split_light_field_model"] = slf_model
    debug["fuzzy_image"] = fuzzy_image
    debug["fuzzy_image_smooth"] = fuzzy_image_smooth
    debug["azimuth_ring"] = azimuth_ring
    debug["azimuth_ring_smooth"] = azimuth_ring_smooth

    return result, debug
