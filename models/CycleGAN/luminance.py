import cv2
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


def extend_triple(channel):
    return np.tile(channel, reps=[3, 1, 1]).transpose(1, 2, 0)


def get_luminance_content_and_style(content, style):
    c_Y, c_I, c_Q = yiq_from_image(content)
    s_Y, s_I, s_Q = yiq_from_image(style)

    s_Y_mean, c_Y_mean = np.mean(s_Y), np.mean(c_Y)
    s_Y_std, c_Y_std = np.std(s_Y), np.std(c_Y)
    # s_Y = (c_Y_std / (s_Y_std + 1e9)) * (s_Y - s_Y_mean) + c_Y_mean

    c_luminance = extend_triple(c_Y) / 255.0
    s_luminance = extend_triple(s_Y) / 255.0

    return (c_luminance, s_luminance)


""" 
path = "./cari_to_pic/train/A/Abdel_Nasser_Assidi_0001_vtoonify_t.jpg"
img = cv2.imread(path)

a, b = get_luminance_content_and_style(img, img)

cv2.imshow("", a)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
