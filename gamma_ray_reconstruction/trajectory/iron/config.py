

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
            "ellipse_model": {"min_num_photons": 3,},
        },
        "core_axis_fit": {
            "c_para": {
                "start_deg": -4.0,
                "stop_deg": 4.0,
                "num_supports": 128,
            },
            "r_para": {"start_m": -640, "stop_m": 640.0, "num_supports": 96,},
            "scan": {"num_bins_radius": 2,},
            "shower_model": {"c_perp_width_deg": 0.1,},
        },
    }