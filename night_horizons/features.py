'''
TODO: Refactor into classes with @staticmethod. Will be more clean.
'''
import copy
import glob
import os
import time
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.validation import check_array
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import utils
from . import raster


class ImageJoiner(utils.LoggerMixin):

    def __init__(
        self,
        feature_detector,
        feature_matcher,
        feature_detector_options={},
        feature_matcher_options={},
        det_min=0.6,
        det_max=2.0,
        dark_frame_brightness=0.03,
        dark_frame_percentile=0.95,
        n_matches_used=500,
        homography_method=cv2.RANSAC,
        reproj_threshold=5.,
        find_homography_options={},
        outline: int = 0,
        debug_mode: bool = False,
        log_keys: list[str] = [],
        fill_empty: bool = False,
    ):

        # Handle feature detector object creation
        if isinstance(feature_detector, str):
            feature_detector_fn = getattr(cv2, f'{feature_detector}_create')
            feature_detector = feature_detector_fn(**feature_detector_options)
        else:
            assert feature_detector_options == {}, \
                'Can only pass options if `feature_detector` is a str'

        # Handle feature matcher object creation
        if isinstance(feature_matcher, str):
            feature_matcher_fn = getattr(cv2, f'{feature_matcher}_create')
            feature_matcher = feature_matcher_fn(**feature_matcher_options)
        else:
            assert feature_matcher_options == {}, \
                'Can only pass options if `feature_matcher` is a str'

        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.det_min = det_min
        self.det_max = det_max
        self.dark_frame_brightness = dark_frame_brightness
        self.dark_frame_percentile = dark_frame_percentile
        self.n_matches_used = n_matches_used
        self.homography_method = homography_method
        self.reproj_threshold = reproj_threshold
        self.find_homography_options = find_homography_options
        self.outline = outline

        # Initialize the log
        super().__init__(debug_mode, log_keys, fill_empty)

    def join(self, src_img, dst_img, warp_and_blend=True):

        start = time.time()

        results = {}
        try:
            # Try to get a valid homography
            results = self.find_valid_homography(src_img, dst_img)

            if warp_and_blend:

                # Warp image
                warped_img = self.warp(src_img, dst_img, results['M'])

                # Blend images
                blended_img = self.blend(
                    warped_img, dst_img, outline=self.outline)

                results['blended_img'] = blended_img

            return_code = 'success'
        except cv2.error:
            return_code = 'opencv_err'
        except HomographyTransformError:
            return_code = 'bad_det'
        except DarkFrameError:
            return_code = 'dark_frame'
        finally:
            duration = time.time() - start
            results['duration'] = duration

            # Log
            self.update_log(locals())

            # DEBUG
            import pdb; pdb.set_trace()

            return return_code, results, self.log

    def find_valid_homography(self, src_img, dst_img):

        results = {}

        # Check for a dark frame
        self.validate_brightness(src_img)

        # Get keypoints
        src_kp, src_des = self.detect_and_compute(src_img)
        dst_kp, dst_des = self.detect_and_compute(dst_img)
        results['src_kp'] = src_kp
        results['src_des'] = src_des

        # Get transform
        M = self.find_homography(
            src_kp,
            src_des,
            dst_kp,
            dst_des,
        )
        results['M'] = M

        # Check transform
        self.validate_homography(M)

        # Log
        self.update_log(locals())

        return results

    def detect_and_compute(self, img):

        return self.feature_detector.detectAndCompute(img, None)

    def find_homography(
        self,
        src_kp,
        src_des,
        dst_kp,
        dst_des,
    ):

        # Perform match
        matches = self.feature_matcher.match(src_des, dst_des)

        # Sort matches in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Take only n_matches_used matches
        matches = matches[:self.n_matches_used]

        # Points for the transform
        src_pts = np.array([src_kp[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2)
        dst_pts = np.array([dst_kp[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2)

        # Get the transform
        M, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=self.homography_method,
            ransacReprojThreshold=self.reproj_threshold,
            **self.find_homography_options
        )

        # Log
        self.update_log(locals())

        return M

    @staticmethod
    def warp(src_img, dst_img, M):

        # Warp the image being fit
        height, width = dst_img.shape[:2]
        warped_img = cv2.warpPerspective(src_img, M, (width, height))

        return warped_img

    @staticmethod
    def warp_bounds(src_img, M):

        bounds = np.array([
            [0., 0.],
            [0., src_img.shape[0]],
            [src_img.shape[1], src_img.shape[0]],
            [src_img.shape[1], 0.],
        ])
        dsframe_bounds = cv2.perspectiveTransform(
            bounds.reshape(-1, 1, 2),
            M,
        ).reshape(-1, 2)

        # Get the new mins and maxs
        px_min, py_min = dsframe_bounds.min(axis=0)
        px_max, py_max = dsframe_bounds.max(axis=0)

        # Put in terms of offset and size
        x_off = px_min
        y_off = py_min
        x_size = px_max - px_min
        y_size = py_max - py_min

        return x_off, y_off, x_size, y_size

    @staticmethod
    def blend(
        src_img,
        dst_img,
        fill_value: Union[float, int] = None,
        outline: int = 0.,
    ):

        # Fill value defaults to values that would be opaque
        if fill_value is None:
            if np.issubdtype(dst_img.dtype, np.integer):
                fill_value = 255
            else:
                fill_value = 1.

        # Doesn't consider zeros in the final channel as empty
        n_bands = dst_img.shape[-1]
        is_empty = (dst_img[:, :, :n_bands - 1].sum(axis=2) == 0)

        # Blend
        blended_img = []
        for j in range(n_bands):
            try:
                blended_img_j = np.where(
                    is_empty,
                    src_img[:, :, j],
                    dst_img[:, :, j]
                )
            # When there's no band information in the one we're blending,
            # fall back to the fill value
            except IndexError:
                blended_img_j = np.full(
                    dst_img.shape[:2],
                    fill_value,
                    dtype=dst_img.dtype
                )
            blended_img.append(blended_img_j)
        blended_img = np.array(blended_img).transpose(1, 2, 0)

        # Add an outline
        if outline > 0:
            blended_img[:outline] = fill_value
            blended_img[-1 - outline:] = fill_value
            blended_img[:, :outline] = fill_value
            blended_img[:, -1 - outline:] = fill_value

        return blended_img

    def validate_brightness(self, img):

        # Get values as fraction of max possible
        values = img.flatten()
        if np.issubdtype(img.dtype, np.integer):
            values = values / np.iinfo(img.dtype).max

        dark_frac = (values < self.dark_frame_brightness).sum() / values.size

        if dark_frac > self.dark_frame_percentile:
            raise DarkFrameError(
                f'Dark frame, dark_frac = {dark_frac:.3g}'
            )

        # Log
        self.update_log(locals())

    def validate_homography(self, M):

        abs_det_M = np.abs(np.linalg.det(M))

        det_in_range = (
            (abs_det_M > self.det_min)
            and (abs_det_M < self.det_max)
        )

        if not det_in_range:
            raise HomographyTransformError(
                f'Bad determinant, abs_det_M = {abs_det_M:.2g}'
            )

        # Log
        self.update_log(locals())


class ImageJoinerQueue:

    def __init__(self, defaults, variations):
        '''
        Example
        -------
        defaults = {
            'feature_detector': 'AKAZE',
            'feature_matcher': 'BFMatcher',
        }
        variations = [
            {'n_matches_used': 100},
            {'n_matches_used': None},
        ]
        image_joiner_queue = ImageJoinerQueue(defaults, variations)
        '''
        self.image_joiners = []
        for var in variations:
            options = copy.deepcopy(defaults)
            options.update(var)
            image_joiner = ImageJoiner(**options)
            self.image_joiners.append(image_joiner)

    def join(self, src_img, dst_img):

        for i, image_joiner in enumerate(self.image_joiners):

            result_code, result, debug_log = image_joiner.join(
                src_img,
                dst_img
            )
            if result_code == 'success':
                break

        debug_log['i_image_joiner'] = i
        return result_code, result, debug_log


class HomographyTransformError(ValueError):
    pass


class DarkFrameError(ValueError):
    pass
