import numpy as np


def rgb_to_yiq(R, G, B):
    Y = 0.229 * R + 0.587 * G + 0.114 * B
    I = 0.595716 * R - 0.274453 * G - 0.321263 * B
    Q = 0.221456 * R - 0.522591 * G + 0.311135 * B

    return Y, I, Q


def yiq_to_rgb(Y, I, Q):
    R = Y + 0.9563 * I + 0.621 * Q
    G = Y - 0.2721 * I - 0.6474 * Q
    B = Y - 1.1070 * I + 1.7046 * Q

    return R, G, B


def yiq_from_image(img):
    B = img[..., 0]
    G = img[..., 1]
    R = img[..., 2]
    Y, I, Q = rgb_to_yiq(R, G, B)

    return Y, I, Q


def extend_to_three_channel(channel):
    return np.tile(channel, reps=[3, 1, 1]).transpose(1, 2, 0)
