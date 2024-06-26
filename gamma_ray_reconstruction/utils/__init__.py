from . import discrete_kernel
import numpy as np


def squarespace(start, stop, num):
    sqrt_space = np.linspace(
        np.sign(start) * np.sqrt(np.abs(start)),
        np.sign(stop) * np.sqrt(np.abs(stop)),
        num,
    )
    signs = np.sign(sqrt_space)
    square_space = sqrt_space**2
    square_space *= signs
    return square_space


def angle_between(v1, v2):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def argmax2d(a):
    return np.unravel_index(np.argmax(a), a.shape)
