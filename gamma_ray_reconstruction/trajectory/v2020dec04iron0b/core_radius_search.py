from ... import utils
from . import shower_model
import numpy as np


def matching_core_radius(c_para, epsilon, m):
    """
    For a given angle between and  (epsilon), and a given distance between
    the aperture' center and the shower's maximum, there is only one
    matching radial distance towards the shower's core along the
    aperture-plane.
    """
    rrr = c_para - 0.5 * np.pi + epsilon
    out = m * (np.cos(epsilon) + np.sin(epsilon) * np.tan(rrr))
    return -1.0 * out


def make_search_mask_for_c_para_r_para(
    config, epsilon, distance_aperture_center_to_shower_maximum
):
    c_para_r_para_mask = np.zeros(
        shape=(
            config["c_para"]["num_supports"],
            config["r_para"]["num_supports"],
        ),
        dtype=int,
    )

    for cbin, c_para in enumerate(config["c_para"]["supports"]):
        matching_r_para = matching_core_radius(
            c_para=c_para,
            epsilon=epsilon,
            m=distance_aperture_center_to_shower_maximum,
        )

        closest_r_para_bin = np.argmin(
            np.abs(config["r_para"]["supports"] - matching_r_para)
        )

        if closest_r_para_bin > 0 and closest_r_para_bin < (
            config["r_para"]["num_supports"] - 1
        ):
            if config["scan"]["num_bins_radius"] == 0:
                rbin_range = [closest_r_para_bin]
            else:
                rbin_range = np.arange(
                    closest_r_para_bin - config["scan"]["num_bins_radius"],
                    closest_r_para_bin + config["scan"]["num_bins_radius"],
                )

            for rbin in rbin_range:
                if rbin >= 0 and rbin < config["r_para"]["num_supports"]:
                    c_para_r_para_mask[cbin, rbin] = 1

    return c_para_r_para_mask


def estimate_core_radius_using_shower_model(
    main_axis_support_cx,
    main_axis_support_cy,
    main_axis_azimuth,
    light_field_cx,
    light_field_cy,
    light_field_x,
    light_field_y,
    shower_maximum_cx,
    shower_maximum_cy,
    shower_maximum_object_distance,
    config,
):
    # mask c_para r_para
    # ------------------

    shower_median_direction_z = np.sqrt(
        1.0 - shower_maximum_cx**2 - shower_maximum_cy**2
    )
    distance_aperture_center_to_shower_maximum = (
        shower_maximum_object_distance / shower_median_direction_z
    )

    shower_maximum_direction = [
        shower_maximum_cx,
        shower_maximum_cy,
        shower_median_direction_z,
    ]

    core_axis_direction = [
        np.cos(main_axis_azimuth),
        np.sin(main_axis_azimuth),
        0.0,
    ]

    epsilon = utils.angle_between(
        shower_maximum_direction, core_axis_direction
    )

    c_para_r_para_mask = make_search_mask_for_c_para_r_para(
        config=config,
        epsilon=epsilon,
        distance_aperture_center_to_shower_maximum=distance_aperture_center_to_shower_maximum,
    )

    # populate c_para r_para
    # ----------------------
    c_para_r_para_response = np.zeros(
        shape=(
            config["c_para"]["num_supports"],
            config["r_para"]["num_supports"],
        )
    )
    for cbin, c_para in enumerate(config["c_para"]["supports"]):
        for rbin, r_para in enumerate(config["r_para"]["supports"]):
            if c_para_r_para_mask[cbin, rbin]:
                c_para_r_para_response[cbin, rbin] = shower_model.response(
                    main_axis_azimuth=main_axis_azimuth,
                    main_axis_support_cx=main_axis_support_cx,
                    main_axis_support_cy=main_axis_support_cy,
                    light_field_cx=light_field_cx,
                    light_field_cy=light_field_cy,
                    light_field_x=light_field_x,
                    light_field_y=light_field_y,
                    c_para=c_para,
                    r_para=r_para,
                    cer_perp_distance_threshold=config["shower_model"][
                        "c_perp_width"
                    ],
                )

    # find highest response in c_para r_para
    # --------------------------------------
    argmax_c_para, argmax_r_para = utils.argmax2d(c_para_r_para_response)
    max_c_para = config["c_para"]["supports"][argmax_c_para]
    max_r_para = config["r_para"]["supports"][argmax_r_para]
    max_response = c_para_r_para_response[argmax_c_para, argmax_r_para]

    # store finding
    # -------------

    reco_cx = main_axis_support_cx + np.cos(main_axis_azimuth) * max_c_para
    reco_cy = main_axis_support_cy + np.sin(main_axis_azimuth) * max_c_para
    reco_x = np.cos(main_axis_azimuth) * max_r_para
    reco_y = np.sin(main_axis_azimuth) * max_r_para

    result = {}

    result["c_main_axis_parallel"] = float(max_c_para)
    result["r_main_axis_parallel"] = float(max_r_para)
    result["shower_model_response"] = float(max_response)

    result["primary_particle_cx"] = float(reco_cx)
    result["primary_particle_cy"] = float(reco_cy)
    result["primary_particle_x"] = float(reco_x)
    result["primary_particle_y"] = float(reco_y)

    debug = {}
    debug["c_para_r_para_mask"] = c_para_r_para_mask
    debug["c_para_r_para_response"] = c_para_r_para_response
    debug["shower_maximum_direction"] = shower_maximum_direction
    debug["core_axis_direction"] = core_axis_direction
    debug["epsilon"] = epsilon

    return result, debug
