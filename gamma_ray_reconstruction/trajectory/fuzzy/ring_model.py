import numpy as np


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
