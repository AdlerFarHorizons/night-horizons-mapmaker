from abc import ABC, abstractmethod
import copy
import glob
import os
import time
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import psutil
import scipy

# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from .. import utils, exceptions

# Set up the logger
LOGGER = utils.get_logger(__name__)


class BaseImageOperator(utils.LoggerMixin, ABC):

    @abstractmethod
    def operate(self, src_img, dst_img):
        pass


class ImageBlender(BaseImageOperator):

    def __init__(
        self,
        fill_value: Union[float, int] = None,
        outline: int = 0.,
        log_keys: list[str] = [],
    ):

        self.fill_value = fill_value
        self.outline = outline
        self.log_keys = log_keys

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:

        # Resize the source image
        src_img_resized = cv2.resize(
            src_img,
            (dst_img.shape[1], dst_img.shape[0]),
        )

        blended_img = self.blend(
            src_img=src_img_resized,
            dst_img=dst_img,
        )

        return {'blended_image': blended_img}

    def blend(
        self,
        src_img: np.ndarray,
        dst_img: np.ndarray,
    ) -> np.ndarray:

        # Fill value defaults to values that would be opaque
        if self.fill_value is None:
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
        if self.outline > 0:
            blended_img[:self.outline] = fill_value
            blended_img[-1 - self.outline:] = fill_value
            blended_img[:, :self.outline] = fill_value
            blended_img[:, -1 - self.outline:] = fill_value

        self.update_log(locals())

        return blended_img


class ImageAligner(BaseImageOperator):

    def __init__(
        self,
        feature_detector,
        feature_matcher,
        image_transformer,
        det_min=0.6,
        det_max=2.0,
        required_brightness=0.03,
        required_bright_pixel_area=50000,
        n_matches_used=500,
        homography_method=cv2.RANSAC,
        reproj_threshold=5.,
        find_homography_options={},
        log_keys: list[str] = ['abs_det_M', 'duration'],
    ):

        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.image_transformer = image_transformer
        self.det_min = det_min
        self.det_max = det_max
        self.required_brightness = required_brightness
        self.required_bright_pixel_area = required_bright_pixel_area
        self.n_matches_used = n_matches_used
        self.homography_method = homography_method
        self.reproj_threshold = reproj_threshold
        self.find_homography_options = find_homography_options
        self.log_keys = log_keys

    def operate(self, src_img, dst_img):

        src_img_t, dst_img_t = self.image_transformer.fit_transform(
            [src_img, dst_img])

        LOGGER.info('Finding homography...')

        # Try to get a valid homography
        results = self.find_valid_homography(src_img_t, dst_img_t)

        LOGGER.info('Warping image...')

        # Warp image
        warped_img = self.warp(src_img, dst_img, results['M'])
        warped_bounds = self.warp_bounds(src_img, results['M'])
        self.validate_warp(dst_img, *warped_bounds)

        results['warped_image'] = warped_img
        results['warped_bounds'] = warped_bounds

        return results

    def apply_img_transform(self, img):

        if self.img_transform is None:
            return img

        return self.img_transform(img)

    def find_valid_homography(self, src_img, dst_img):
        '''
        Parameters
        ----------
        Returns
        -------
            results:
                M: Homography transform.
                src_kp: Keypoints for the src image.
                src_des: Keypoint descriptors for the src image.
        '''

        # Check what's in bounds, exit if nothing
        if dst_img.sum() == 0:
            raise exceptions.OutOfBoundsError('No dst data in bounds.')

        results = {}

        LOGGER.info('Validating brightness...')

        # Check for a dark frame
        self.validate_brightness(src_img)
        self.validate_brightness(dst_img, error_type='dst')

        LOGGER.info('Detecting and computing keypoints...')

        mem = psutil.virtual_memory()
        LOGGER.info(
            'Memory status: '
            f'{mem.available / 1024**3.:.2g} of '
            f'{mem.total / 1024**3.:.2g} GB available '
            f'({mem.percent}% used)'
        )

        # Get keypoints
        src_kp, src_des = self.detect_and_compute(src_img)
        # TODO: This is a point at which the process can be killed
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

        LOGGER.info('Validating homography...')

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

        LOGGER.info('Matching...')

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

        LOGGER.info('Finding transform...')

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
        '''Warp the bounds of the source image to get the bounds of the
        warped image in the frame of the destination image.

        Parameters
        ----------
        Returns
        -------
        '''

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

    def validate_warp(self, dst_img, x_off, y_off, x_size, y_size):
        '''The viable range in the destination image frame is
        (0, dst_img.shape[1]) in the x-direction, and (0, dst_img.shape[0])
        in the y direction.

        Parameters
        ----------
        Returns
        -------
        '''

        if (
            (x_off < 0)
            | (y_off < 0)
            | (x_off + x_size > dst_img.shape[1])
            | (y_off + y_size > dst_img.shape[0])
        ):
            raise exceptions.OutOfBoundsError(
                'Warping results in out-of-bounds image'
            )

    def validate_brightness(self, img, error_type='src'):

        # Get values as fraction of max possible
        values = img.flatten()
        if np.issubdtype(img.dtype, np.integer):
            values = values / np.iinfo(img.dtype).max

        bright_area = (values > self.required_brightness).sum()

        # Log
        bright_frac = bright_area / values.size
        req_bright_frac = bright_area / values.size
        self.update_log(locals())

        if bright_area < self.required_bright_pixel_area:
            if error_type == 'src':
                error_type = exceptions.SrcDarkFrameError
            elif error_type == 'dst':
                error_type == exceptions.DstDarkFrameError
            else:
                raise KeyError(
                    'Unrecognized error type in validate_brightness')

            raise error_type(
                'Insufficient bright pixels to calculate features. '
                f'Have {bright_area} pixels^2, '
                f'i.e. a bright frac of {bright_frac:.3g}. '
                f'Need {self.required_bright_pixel_area} pixels^2, '
                f'i.e. a bright frac of {req_bright_frac:.3g}.'
            )

    def validate_homography(self, M):

        if M is None:
            raise exceptions.HomographyTransformError(
                'Transform matrix M is None'
            )

        abs_det_M = np.abs(np.linalg.det(M))

        det_in_range = (
            (abs_det_M > self.det_min)
            and (abs_det_M < self.det_max)
        )

        # Log
        self.update_log(locals())

        if not det_in_range:
            raise exceptions.HomographyTransformError(
                f'Bad determinant, abs_det_M = {abs_det_M:.2g}'
            )


class ImageAlignerBlender(ImageAligner, ImageBlender):

    def __init__(
        self,
        feature_detector,
        feature_matcher,
        image_transformer,
        det_min=0.6,
        det_max=2.0,
        required_brightness=0.03,
        required_bright_pixel_area=50000,
        n_matches_used=500,
        homography_method=cv2.RANSAC,
        reproj_threshold=5.,
        find_homography_options={},
        fill_value: Union[float, int] = None,
        outline: int = 0,
        log_keys: list[str] = ['abs_det_M', 'duration'],
    ):

        super().__init__(
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            image_transformer=image_transformer,
            det_min=det_min,
            det_max=det_max,
            required_brightness=required_brightness,
            required_bright_pixel_area=required_bright_pixel_area,
            n_matches_used=n_matches_used,
            homography_method=homography_method,
            reproj_threshold=reproj_threshold,
            find_homography_options=find_homography_options,
        )

        super(ImageAligner, self).__init__(
            fill_value=fill_value,
            outline=outline,
            log_keys=log_keys,
        )

    def operate(self, src_img, dst_img):

        LOGGER.info('Aligning image...')

        # Align images
        align_results = super().operate(src_img, dst_img)

        LOGGER.info('Blending images...')

        # Blend images
        blend_results = super(ImageAligner, self).operate(
            align_results['warped_image'], dst_img)

        results = {**align_results, **blend_results}

        return results


class ImageProcessorQueue:

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
            image_joiner = ImageAlignerBlender(**options)
            self.image_joiners.append(image_joiner)

    def operate(self, src_img, dst_img):

        for i, image_joiner in enumerate(self.image_joiners):

            result_code, result, log = image_joiner.process(
                src_img,
                dst_img
            )
            if result_code == 'success':
                break

        log['i_image_joiner'] = i
        return result_code, result, log
