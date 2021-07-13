import numpy as np
from skimage.draw import line_aa as skimage_draw_line_aa


"""
Estimate model for each image-sequence
--------------------------------------
"""


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
    for pax in range(split_light_field.number_paxel):
        img = split_light_field.image_sequences[pax]
        num_photons = img.shape[0]
        if num_photons >= model_config["min_num_photons"]:
            models.append(
                estimate_model_from_image_sequence(
                    cx=img[:, 0], cy=img[:, 1], t=img[:, 2]
                )
            )
    return models


"""
Create fuzzy-image from models
------------------------------
"""


def _draw_line(r0, c0, r1, c1, image_shape):
    rr, cc, aa = skimage_draw_line_aa(r0=r0, c0=c0, r1=r1, c1=c1)
    valid_rr = np.logical_and((rr >= 0), (rr < image_shape[0]))
    valid_cc = np.logical_and((cc >= 0), (cc < image_shape[1]))
    valid = np.logical_and(valid_rr, valid_cc)
    return rr[valid], cc[valid], aa[valid]


def draw_line_model(model, image_binning):
    fov_radius = image_binning["radius"]
    pix_per_rad = image_binning["num_bins"] / (2.0 * fov_radius)
    image_middle_px = image_binning["num_bins"] // 2

    cen_x = model["median_cx"]
    cen_y = model["median_cy"]

    off_x = fov_radius * np.sin(model["azimuth"])
    off_y = fov_radius * np.cos(model["azimuth"])
    start_x = cen_x + off_x
    start_y = cen_y + off_y
    stop_x = cen_x - off_x
    stop_y = cen_y - off_y

    start_x_px = int(np.round(start_x * pix_per_rad)) + image_middle_px
    start_y_px = int(np.round(start_y * pix_per_rad)) + image_middle_px

    stop_x_px = int(np.round(stop_x * pix_per_rad)) + image_middle_px
    stop_y_px = int(np.round(stop_y * pix_per_rad)) + image_middle_px

    rr, cc, aa = _draw_line(
        r0=start_y_px,
        c0=start_x_px,
        r1=stop_y_px,
        c1=stop_x_px,
        image_shape=(image_binning["num_bins"], image_binning["num_bins"]),
    )

    return rr, cc, aa


def make_image_from_model(split_light_field_model, image_binning):
    out = np.zeros(
        shape=(image_binning["num_bins"], image_binning["num_bins"])
    )
    for model in split_light_field_model:
        rr, cc, aa = draw_line_model(
            model=model, image_binning=image_binning
        )
        out[rr, cc] += aa * model["num_photons"]
    return out


"""
analyse fuzzy image
-------------------
"""

def argmax2d(a):
    return np.unravel_index(np.argmax(a), a.shape)


def argmax_image_cx_cy(image, image_binning):
    _resp = argmax2d(image)
    reco_cx = image_binning["c_bin_centers"][_resp[1]]
    reco_cy = image_binning["c_bin_centers"][_resp[0]]
    return reco_cx, reco_cy


def project_image_onto_ring(
    image,
    image_binning,
    ring_cx,
    ring_cy,
    ring_radius,
    ring_binning,
):
    pix_per_rad = image_binning["num_bins"] / (2.0 * image_binning["radius"])
    image_middle_px = image_binning["num_bins"] // 2

    ring = np.zeros(ring_binning["num_bins"])
    for ia, az in enumerate(ring_binning["bin_edges"]):

        for probe_radius in np.linspace(ring_radius / 2, ring_radius, 5):
            probe_cx = ring_cx + np.cos(az) * probe_radius
            probe_cy = ring_cy + np.sin(az) * probe_radius

            probe_x_px = int(probe_cx * pix_per_rad + image_middle_px)
            probe_y_px = int(probe_cy * pix_per_rad + image_middle_px)
            valid_x = np.logical_and(
                probe_x_px >= 0, probe_x_px < image_binning["num_bins"]
            )
            valid_y = np.logical_and(
                probe_y_px >= 0, probe_y_px < image_binning["num_bins"]
            )
            if valid_x and valid_y:
                ring[ia] += image[probe_y_px, probe_x_px]

    return ring


def circular_convolve1d(in1, in2):
    work = np.concatenate([in1, in1, in1])
    work_conv = np.convolve(work, in2, mode="same")
    return work_conv[in1.shape[0] : 2 * in1.shape[0]]


def circular_argmaxima(ring):
    work = np.concatenate([ring, ring, ring])
    work_gradient = np.gradient(work)
    signchange = np.zeros(3 * ring.shape[0], dtype=int)
    for ii in range(3 * ring.shape[0] - 2):
        a0 = work_gradient[ii]
        a1 = work_gradient[ii + 2]
        if a0 > 0.0 and a1 < 0.0:
            signchange[ii + 1] = 1
    rang = signchange[ring.shape[0] : 2 * ring.shape[0]]
    return np.where(rang)[0]
