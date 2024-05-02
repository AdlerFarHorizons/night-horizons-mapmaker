from abc import ABC, abstractmethod
import copy
from typing import Union

import cv2
import numpy as np
import psutil

from .. import utils, exceptions
from ..transformers.raster import BaseImageTransformer

# Set up the logger
LOGGER = utils.get_logger(__name__)


class BaseImageOperator(utils.LoggerMixin, ABC):
    '''Base class for image operators—classes that perform an operation on two images.
    '''

    @abstractmethod
    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
            '''Abstract method for performing an operation on two images.

            Parameters
            ----------
            src_img : numpy.ndarray
                The source image on which the operation will be performed.
            dst_img : numpy.ndarray
                The destination image where the result of the operation will be stored.

            Returns
            -------
            dict
                Results dictionary.
            '''
            pass


class ImageBlender(BaseImageOperator):
    '''Class for combining two images by filling empty space in the dst image
    with the src image. No averaging is performed.
    '''

    def __init__(
        self,
        fill_value: Union[float, int] = None,
        outline: int = 0,
        log_keys: list[str] = [],
    ):
        '''
        Initialize the ImageBlender object.

        Parameters
        ----------
        fill_value : Union[float, int], optional
            The value used to fill empty pixels in the image. Defaults to values
            that are maximum for the band.
        outline : int, optional
            The width of the outline to draw around the blended image. Good for
            checking how images are combined.
        log_keys : list[str], optional
            A list of internal variables to log the values of.
            Defaults to an empty list.
        '''
        self.fill_value = fill_value
        self.outline = outline
        self.log_keys = log_keys

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        '''Perform the blending operation.

        This method takes in a source image and a destination image, and performs
        the blending operation. It resizes the source image to match the dimensions
        of the destination image, and then blends the two images together using the
        `blend` method.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to be blended.

        dst_img : np.ndarray
            The destination image.

        Returns
        -------
        dict
            A dictionary containing the blended image.
        '''

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
        '''Blend images together, filling empty space in the destination image
        with the source image.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to blend into the destination image.
        dst_img : np.ndarray
            The destination image where the source image will be blended.

        Returns
        -------
        np.ndarray
            The blended image.
        '''

        # Fill value defaults to values that would be opaque
        if self.fill_value is None:
            if np.issubdtype(dst_img.dtype, np.integer):
                fill_value = 255
            else:
                fill_value = 1.

        # Find where the sum of the bands is zero, i.e. the image is empty.
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
    '''Class for aligning a source image with a destination image based on
    the features detected in the images.
    '''

    def __init__(
        self,
        feature_detector: cv2.Feature2D,
        feature_matcher: cv2.DescriptorMatcher,
        image_transformer: BaseImageTransformer,
        det_min: float = 0.6,
        det_max: float = 1.7,
        required_brightness: float = 0.03,
        required_bright_pixel_area: int = 50000,
        n_matches_used: int = 500,
        find_homography_options: dict = {
            'method': cv2.RANSAC,
            'ransacReprojThreshold': 5.0,
        },
        log_keys: list[str] = ['abs_det_M', 'duration'],
    ):
        '''
        Initializes the Operator object.

        Parameters
        ----------
        feature_detector : cv2.Feature2D
            The feature detector used for detecting keypoints in images.
        feature_matcher : cv2.DescriptorMatcher
            The feature matcher used for matching keypoints between images.
        image_transformer : BaseImageTransformer
            The image transformer used for transforming images before detection
            and matching. Can be used to apply a logscale transformation for example.
        det_min : float, optional
            The minimum determinant value for accepting a homography matrix,
            by default 0.6. Values less than this indicate a highly warped image,
            consistent with a bad homography.
        det_max : float, optional
            The maximum determinant value for accepting a homography matrix,
            by default 1.7. Values greater than this indicate a highly warped image,
            consistent with a bad homography.
        required_brightness : float, optional
            The minimum value for a given pixel to be considered as having data
            (as opposed to being dark) as a fraction of the maximum possible value.
            Default is 0.03.
        required_bright_pixel_area : int, optional
            The required area in square pixels of bright pixels for an image not to be
            a dark frame., Default is 50000.
        n_matches_used : int, optional
            The number of matches used for computing the homography matrix,
            by default 500.
        find_homography_options : dict, optional
            Options passed to the findHomography function. More info in the docstring
            for opencv's findHomography function. Default is RANSAC with a reprojection
            threshold of 5.
        log_keys : list[str], optional
            Local variables to be logged during the operator execution,
            by default ['abs_det_M', 'duration'].
        '''

        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.image_transformer = image_transformer
        self.det_min = det_min
        self.det_max = det_max
        self.required_brightness = required_brightness
        self.required_bright_pixel_area = required_bright_pixel_area
        self.n_matches_used = n_matches_used
        self.find_homography_options = find_homography_options
        self.log_keys = log_keys

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        '''Align the src_img with the dst_img.

        Parameters
        ----------
        src_img : numpy.ndarray
            The source image to be aligned.

        dst_img : numpy.ndarray
            The destination image to align the source image with.

        Returns
        -------
        dict
            The results dictionary containing the following keys:
            - 'warped_image': The warped source image aligned with the
                destination image.
            - 'warped_bounds': The bounding box of the warped image.

        '''

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

    def find_valid_homography(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        '''Find a valid homography transform between the source image and
        the destination image. Raises an error if there is no valid homography.

        Parameters
        ----------
        src_img : np.ndarray
            The source image.
        dst_img : np.ndarray
            The destination image.

        Returns
        -------
        dict
            A dictionary containing the following results:
            - M : Homography transform.
            - src_kp : Keypoints for the source image.
            - src_des : Keypoint descriptors for the source image.
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
        LOGGER.info('Detecting and computing keypoints:source...')
        src_kp, src_des = self.detect_and_compute(src_img)
        LOGGER.info('Detecting and computing keypoints:destination...')
        dst_kp, dst_des = self.detect_and_compute(dst_img)
        results['src_kp'] = src_kp
        results['src_des'] = src_des

        LOGGER.info('Finding homography....')

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

    def detect_and_compute(
        self,
        img: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], list[np.ndarray]]:
        '''Detect and compute feature-matching keypoints in an image,
        as well as their descriptors.

        Parameters
        ----------
        img: np.ndarray
            Input image to detect keypoitns for.

        Returns
        -------
        tuple[list[cv2.KeyPoint], list[np.ndarray]]:
            The detected keypoints and their descriptors.
        '''

        return self.feature_detector.detectAndCompute(img, None)

    def find_homography(
        self,
        src_kp: list[cv2.KeyPoint],
        src_des: list[np.ndarray],
        dst_kp: list[cv2.KeyPoint],
        dst_des: list[np.ndarray],
    ) -> np.ndarray:
        '''Find the homography transform between the source and destination images.

        Parameters
        ----------
        src_kp : list[cv2.KeyPoint]
            List of keypoints in the source image.
        src_des : list[np.ndarray]
            List of descriptors in the source image.
        dst_kp : list[cv2.KeyPoint]
            List of keypoints in the destination image.
        dst_des : list[np.ndarray]
            List of descriptors in the destination image.

        Returns
        -------
        np.ndarray
            The homography matrix representing the transformation between
            the source and destination images.
        '''

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
            **self.find_homography_options
        )

        # Log
        self.update_log(locals())

        return M

    @staticmethod
    def warp(src_img: np.ndarray, dst_img: np.ndarray, M: np.ndarray) -> np.ndarray:
        '''
        Warp the source image using a perspective transformation matrix.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to be warped.

        dst_img : np.ndarray
            The destination image where the warped image will be placed.

        M : np.ndarray
            The perspective transformation matrix.

        Returns
        -------
        np.ndarray
            The warped image.
        '''

        # Warp the image being fit
        height, width = dst_img.shape[:2]
        warped_img = cv2.warpPerspective(src_img, M, (width, height))

        return warped_img

    @staticmethod
    def warp_bounds(src_img: np.ndarray, M: np.ndarray) -> list[int]:
        '''Warp the bounds of the source image to get the bounds of the
        warped image in the frame of the destination image.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to be warped.

        M : np.ndarray
            The transformation matrix used for warping.

        Returns
        -------
        list[int]
            A list containing the x offset, y offset, width, and height of the
            warped image in the frame of the destination image.
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

    def validate_warp(
        self,
        dst_img: np.ndarray,
        x_off: int,
        y_off: int,
        x_size: int,
        y_size: int
    ):
        '''Check if a warp is valid.

        This method checks if the specified warp is valid by ensuring that the
        destination image frame is not exceeded. The viable range in the destination
        image frame is (0, dst_img.shape[1]) in the x-direction, and (0, dst_img.shape[0])
        in the y-direction.

        Parameters
        ----------
        dst_img : np.ndarray
            The destination image frame.
        x_off : int
            The x-coordinate offset of the warp.
        y_off : int
            The y-coordinate offset of the warp.
        x_size : int
            The width of the warp.
        y_size : int
            The height of the warp.

        Raises
        ------
        exceptions.OutOfBoundsError
            If the warp results in an out-of-bounds image.
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

    def validate_brightness(self, img: np.ndarray, error_type: str = 'src'):
        '''Check if the image is bright enough to perform feature matching on.

        Parameters
        ----------
        img : np.ndarray
            The input image to be validated.

        error_type : str, optional
            The type of error to raise if the image is not bright enough (i.e. is
            this an error with the source or the destination).Default is 'src'.

        Raises
        ------
        exceptions.SrcDarkFrameError or exceptions.DstDarkFrameError or KeyError
            If the image does not meet the required brightness criteria.
        '''

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

    def validate_homography(self, M: np.ndarray):
        '''Check if the homography matrix is valid, i.e. it will not result in extreme
        warping when applied.

        Parameters
        ----------
        M : np.ndarray
            The homography matrix to be validated.

        Raises
        ------
        HomographyTransformError
            If the transform matrix M is None or if the determinant of M is outside the
            specified range.
        '''

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
    '''Class for first aligning a src image with a dst image, and then blending
    the two.'''

    def __init__(
        self,
        feature_detector: cv2.Feature2D,
        feature_matcher: cv2.DescriptorMatcher,
        image_transformer: BaseImageTransformer,
        det_min: float = 0.6,
        det_max: float = 1.7,
        required_brightness: float = 0.03,
        required_bright_pixel_area: int = 50000,
        n_matches_used: int = 500,
        find_homography_options: dict = {
            'method': cv2.RANSAC,
            'ransacReprojThreshold': 5.0,
        },
        fill_value: Union[float, int] = None,
        outline: int = 0,
        log_keys: list[str] = ['abs_det_M', 'duration'],
    ):
        '''
        Initializes the ImageAlignerBlender.

        Parameters
        ----------
        feature_detector : cv2.Feature2D
            The feature detector used for detecting keypoints in images.
        feature_matcher : cv2.DescriptorMatcher
            The feature matcher used for matching keypoints between images.
        image_transformer : BaseImageTransformer
            The image transformer used for transforming images before detection
            and matching. Can be used to apply a logscale transformation for example.
        det_min : float, optional
            The minimum determinant value for accepting a homography matrix,
            by default 0.6. Values less than this indicate a highly warped image,
            consistent with a bad homography.
        det_max : float, optional
            The maximum determinant value for accepting a homography matrix,
            by default 1.7. Values greater than this indicate a highly warped image,
            consistent with a bad homography.
        required_brightness : float, optional
            The minimum value for a given pixel to be considered as having data
            (as opposed to being dark) as a fraction of the maximum possible value.
            Default is 0.03.
        required_bright_pixel_area : int, optional
            The required area in square pixels of bright pixels for an image not to be
            a dark frame., Default is 50000.
        n_matches_used : int, optional
            The number of matches used for computing the homography matrix,
            by default 500.
        find_homography_options : dict, optional
            Options passed to the findHomography function. More info in the docstring
            for opencv's findHomography function. Default is RANSAC with a reprojection
            threshold of 5.
        fill_value : Union[float, int], optional
            The value used to fill empty pixels in the image. Defaults to values
            that are maximum for the band.
        outline : int, optional
            The width of the outline to draw around the blended image. Good for
            checking how images are combined.
        log_keys : list[str], optional
            Local variables to be logged during the operator execution,
            by default ['abs_det_M', 'duration'].
        '''

        super().__init__(
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            image_transformer=image_transformer,
            det_min=det_min,
            det_max=det_max,
            required_brightness=required_brightness,
            required_bright_pixel_area=required_bright_pixel_area,
            n_matches_used=n_matches_used,
            find_homography_options=find_homography_options,
        )

        super(ImageAligner, self).__init__(
            fill_value=fill_value,
            outline=outline,
            log_keys=log_keys,
        )

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        '''
        Perform image alignment and blending.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to be aligned and blended.
        dst_img : np.ndarray
            The destination image onto which the source image will be blended.

        Returns
        -------
        dict
            A dictionary containing the results of the image alignment and blending.
        '''

        LOGGER.info('Aligning image...')

        # Align images
        align_results = super().operate(src_img, dst_img)

        LOGGER.info('Blending images...')

        # Blend images
        blend_results = super(ImageAligner, self).operate(
            align_results['warped_image'], dst_img)

        results = {**align_results, **blend_results}

        return results


class ImageOperatorQueue:
    '''Class for operating on images using a queue of image operators.
    The first image operator in the queue attempts to analyze one image.
    If it fails, the next image operator in the queue is used, and so on.
    '''

    def __init__(self, defaults: dict, variations: list[dict]):
        '''Construct the ImageOperatorQueue object.

        Parameters
        ----------
        defaults : dict
            A dictionary containing the default options for the image operator.

        variations : list[dict]
            A list of dictionaries, where each dictionary represents a variation
            of the image operator options. Each dictionary should contain the
            keys that need to be overridden from the default options.

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
        image_operator_queue = ImageOperatorQueue(defaults, variations)
        '''
        self.image_joiners = []
        for var in variations:
            options = copy.deepcopy(defaults)
            options.update(var)
            image_joiner = ImageAlignerBlender(**options)
            self.image_joiners.append(image_joiner)

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        '''Perform the operations in order, until one succeeds.

        Parameters
        ----------
        src_img : np.ndarray
            The source image on which the operations will be performed.
        dst_img : np.ndarray
            The destination image where the result of the operations will be stored.

        Returns
        -------
        tuple
            A tuple containing the result code, the result image, and a log dictionary.
        '''

        for i, image_joiner in enumerate(self.image_joiners):

            result_code, result, log = image_joiner.process(
                src_img,
                dst_img
            )
            if result_code == 'success':
                break

        log['i_image_joiner'] = i
        return result_code, result, log
