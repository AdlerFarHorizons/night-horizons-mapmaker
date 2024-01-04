import cv2
import numpy as np

from .operators import BaseImageOperator


class SimilarityScorer(BaseImageOperator):

    def __init__(
        self,
        allow_resize: bool = True,
        compare_nonzero: bool = True,
        tm_metric=cv2.TM_CCOEFF_NORMED
    ):
        self.allow_resize = allow_resize
        self.compare_nonzero = compare_nonzero
        self.tm_metric = tm_metric

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:

        if src_img.shape != dst_img.shape:
            if not self.allow_resize:
                raise ValueError('Images must have the same shape.')
            src_img = cv2.resize(src_img, (dst_img.shape[1], dst_img.shape[0]))

        if self.compare_nonzero:
            empty_src = np.isclose(src_img.sum(axis=2), 0.)
            empty_dst = np.isclose(dst_img.sum(axis=2), 0.)
            either_empty = np.logical_or(empty_src, empty_dst)
            src_img[either_empty] = 0
            dst_img[either_empty] = 0

        r = cv2.matchTemplate(src_img, dst_img, self.tm_metric)[0][0]

        return {'score': r}
