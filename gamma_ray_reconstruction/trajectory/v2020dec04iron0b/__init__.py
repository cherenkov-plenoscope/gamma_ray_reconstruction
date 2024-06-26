from . import config
from . import shower_model
from . import core_radius_search
from .ShowerModelFit import ShowerModelFit
from .. import v2020nov12fuzzy0


import numpy as np
import plenopy as pl
import iminuit


def estimate(
    loph_record,
    light_field_geometry,
    shower_maximum_object_distance,
    fuzzy_config,
    model_fit_config,
):
    """
    Reconstruct the gamma-ray trajectory w.r.t. to the plenoscope
    """
    lfg = light_field_geometry

    fuzzy_result, fuzzy_debug = v2020nov12fuzzy0.estimate_main_axis_to_core(
        loph_record=loph_record,
        light_field_geometry=light_field_geometry,
        model_config=fuzzy_config["ellipse_model"],
        image_binning=fuzzy_config["image"],
        image_smoothing_kernel=fuzzy_config["image"]["smoothing_kernel"],
        ring_binning=fuzzy_config["azimuth_ring"],
        ring_smoothing_kernel=fuzzy_config["azimuth_ring"]["smoothing_kernel"],
    )

    lixel_ids = loph_record["photons"]["channels"]

    shower_maximum_cx, shower_maximum_cy = pl.split_light_field.median_cx_cy(
        loph_record=loph_record, light_field_geometry=light_field_geometry
    )

    main_axis_to_core_finder = ShowerModelFit(
        light_field_cx=lfg.cx_mean[lixel_ids],
        light_field_cy=lfg.cy_mean[lixel_ids],
        light_field_x=lfg.x_mean[lixel_ids],
        light_field_y=lfg.y_mean[lixel_ids],
        shower_maximum_cx=shower_maximum_cx,
        shower_maximum_cy=shower_maximum_cy,
        shower_maximum_object_distance=shower_maximum_object_distance,
        config=model_fit_config,
    )

    minimizer = iminuit.Minuit(
        fcn=main_axis_to_core_finder.evaluate_shower_model,
        main_axis_azimuth=fuzzy_result["main_axis_azimuth"],
        main_axis_support_perp_offset=0.0,
        # print_level=0,
    )
    minimizer.limits["main_axis_azimuth"] = (
        fuzzy_result["main_axis_azimuth"] - 2.0 * np.pi,
        fuzzy_result["main_axis_azimuth"] + 2.0 * np.pi,
    )
    minimizer.errors["main_axis_azimuth"] = fuzzy_result[
        "main_axis_azimuth_uncertainty"
    ]
    minimizer.limits["main_axis_support_perp_offset"] = (
        -5.0 * fuzzy_result["main_axis_support_uncertainty"],
        5.0 * fuzzy_result["main_axis_support_uncertainty"],
    )
    minimizer.errors["main_axis_support_perp_offset"] = fuzzy_result[
        "main_axis_support_uncertainty"
    ]
    minimizer.errordef = iminuit.Minuit.LEAST_SQUARES
    minimizer.migrad()

    return (
        main_axis_to_core_finder.final_result,
        {
            "fuzzy_result": fuzzy_result,
            "fuzzy_debug": fuzzy_debug,
        },
    )


def is_valid_estimate(estimate):
    keys = [
        "primary_particle_cx",
        "primary_particle_cy",
        "primary_particle_x",
        "primary_particle_y",
    ]
    for key in keys:
        if np.isnan(estimate[key]):
            return False
    return True


def model_response_for_true_trajectory(
    true_cx,
    true_cy,
    true_x,
    true_y,
    loph_record,
    light_field_geometry,
    model_fit_config,
):
    lfg = light_field_geometry

    shower_maximum_cx, shower_maximum_cy = pl.split_light_field.median_cx_cy(
        loph_record=loph_record, light_field_geometry=light_field_geometry
    )

    true_main_axis_azimuth = np.pi + np.arctan2(true_y, true_x)
    true_r_para = np.hypot(true_x, true_y) * np.sign(
        true_main_axis_azimuth - np.pi
    )
    true_c_para = np.hypot(
        shower_maximum_cx - true_cx,
        shower_maximum_cy - true_cy,
    )

    lixel_ids = loph_record["photons"]["channels"]

    true_response = shower_model.response(
        main_axis_azimuth=true_main_axis_azimuth,
        main_axis_support_cx=shower_maximum_cx,
        main_axis_support_cy=shower_maximum_cy,
        light_field_cx=lfg.cx_mean[lixel_ids],
        light_field_cy=lfg.cy_mean[lixel_ids],
        light_field_x=lfg.x_mean[lixel_ids],
        light_field_y=lfg.y_mean[lixel_ids],
        c_para=true_c_para,
        r_para=true_r_para,
        cer_perp_distance_threshold=model_fit_config["shower_model"][
            "c_perp_width"
        ],
    )

    return true_response
