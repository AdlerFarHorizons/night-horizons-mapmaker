import cv2
import numpy as np

R_ACCEPT = 0.75


def image_to_image_ccoeff(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    allow_resize: bool = True,
    compare_nonzero: bool = True,
    tm_metric=cv2.TM_CCOEFF_NORMED
):

    if src_img.shape != dst_img.shape:
        if not allow_resize:
            raise ValueError('Images must have the same shape.')
        src_img = cv2.resize(src_img, (dst_img.shape[1], dst_img.shape[0]))

    if compare_nonzero:
        empty_src = np.isclose(src_img.sum(axis=2), 0.)
        empty_dst = np.isclose(dst_img.sum(axis=2), 0.)
        either_empty = np.logical_or(empty_src, empty_dst)
        src_img[either_empty] = 0
        dst_img[either_empty] = 0

    r = cv2.matchTemplate(src_img, dst_img, tm_metric)[0][0]

    return r


def assert_approx_equal(img1, img2, r_accept=R_ACCEPT):

    r = image_to_image_ccoeff(img1, img2)

    assert r > r_accept
