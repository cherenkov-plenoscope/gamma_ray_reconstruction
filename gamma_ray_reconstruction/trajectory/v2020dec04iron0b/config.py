import numpy as np
from ... import utils


EXAMPLE = {
    "c_para": {
        "start_deg": -4.0,
        "stop_deg": 4.0,
        "num_supports": 128,
    },
    "r_para": {
        "start_m": -640,
        "stop_m": 640.0,
        "num_supports": 96,
    },
    "scan": {
        "num_bins_radius": 2,
    },
    "shower_model": {
        "c_perp_width_deg": 0.1,
    },
}


def compile_user_config(user_config):
    uc = user_config
    cfg = {}
    cfg["c_para"] = {}
    cfg["c_para"]["start"] = np.deg2rad(uc["c_para"]["start_deg"])
    cfg["c_para"]["stop"] = np.deg2rad(uc["c_para"]["stop_deg"])
    cfg["c_para"]["num_supports"] = uc["c_para"]["num_supports"]
    cfg["c_para"]["supports"] = utils.squarespace(
        start=cfg["c_para"]["start"],
        stop=cfg["c_para"]["stop"],
        num=cfg["c_para"]["num_supports"],
    )
    cfg["r_para"] = {}
    cfg["r_para"]["start"] = uc["r_para"]["start_m"]
    cfg["r_para"]["stop"] = uc["r_para"]["stop_m"]
    cfg["r_para"]["num_supports"] = uc["r_para"]["num_supports"]
    cfg["r_para"]["supports"] = utils.squarespace(
        start=cfg["r_para"]["start"],
        stop=cfg["r_para"]["stop"],
        num=cfg["r_para"]["num_supports"],
    )
    cfg["scan"] = dict(uc["scan"])
    cfg["shower_model"] = {}
    cfg["shower_model"]["c_perp_width"] = np.deg2rad(
        uc["shower_model"]["c_perp_width_deg"]
    )
    return cfg


def make_example_config_for_71m_plenoscope(fov_radius_deg):
    return {
        "fuzzy_method": {
            "image": {
                "radius_deg": fov_radius_deg + 1.0,
                "num_bins": 128,
                "smoothing_kernel_width_deg": 0.3125,
            },
            "azimuth_ring": {
                "num_bins": 360,
                "radius_deg": 1.0,
                "smoothing_kernel_width_deg": 41.0,
            },
            "ellipse_model": {
                "min_num_photons": 3,
            },
        },
        "core_axis_fit": EXAMPLE,
    }
