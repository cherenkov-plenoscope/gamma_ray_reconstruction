import numpy as np
import skimage.draw
from ... import utils


def _draw_line(r0, c0, r1, c1, image_shape):
    rr, cc, aa = skimage.draw.line_aa(r0=r0, c0=c0, r1=r1, c1=c1)
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
        rr, cc, aa = draw_line_model(model=model, image_binning=image_binning)
        out[rr, cc] += aa * model["num_photons"]
    return out


"""
analyse fuzzy image
-------------------
"""


def argmax_image_cx_cy(image, image_binning):
    _resp = utils.argmax2d(image)
    reco_cx = image_binning["c_bin_centers"][_resp[1]]
    reco_cy = image_binning["c_bin_centers"][_resp[0]]
    return reco_cx, reco_cy
