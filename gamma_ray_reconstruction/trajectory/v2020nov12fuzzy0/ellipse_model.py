import numpy as np

EXAMPLE_CONFIG = {
    "min_num_photons": 3,
}


def project_onto_main_axis_of_model(cx, cy, ellipse_model):
    ccx = cx - ellipse_model["median_cx"]
    ccy = cy - ellipse_model["median_cy"]
    _cos = np.cos(ellipse_model["azimuth"])
    _sin = np.sin(ellipse_model["azimuth"])
    c_main_axis = ccx * _cos - ccy * _sin
    return c_main_axis


def estimate_ellipse(cx, cy):
    median_cx = np.median(cx)
    median_cy = np.median(cy)

    cov_matrix = np.cov(np.c_[cx, cy].T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_values)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0

    major_axis = eigen_vectors[:, major_idx]
    major_std = np.sqrt(np.abs(eigen_values[major_idx]))
    minor_std = np.sqrt(np.abs(eigen_values[minor_idx]))

    azimuth = np.arctan2(major_axis[0], major_axis[1])
    return {
        "median_cx": median_cx,
        "median_cy": median_cy,
        "azimuth": azimuth,
        "major_std": major_std,
        "minor_std": minor_std,
    }


def estimate_model_from_image_sequence(cx, cy, t):
    assert len(cx) == len(cy)
    assert len(cx) == len(t)

    model = estimate_ellipse(cx=cx, cy=cy)
    model["num_photons"] = float(len(cx))

    c_main_axis = project_onto_main_axis_of_model(
        cx=cx, cy=cy, ellipse_model=model
    )
    return model


def estimate_model_from_light_field(split_light_field, model_config):
    models = []
    for pax in range(split_light_field["number_paxel"]):
        img = split_light_field["image_sequences"][pax]
        num_photons = img.shape[0]
        if num_photons >= model_config["min_num_photons"]:
            models.append(
                estimate_model_from_image_sequence(
                    cx=img[:, 0], cy=img[:, 1], t=img[:, 2]
                )
            )
    return models
