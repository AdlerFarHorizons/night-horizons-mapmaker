import cv2
import numpy as np
import pandas as pd
import pyproj

from .operators import BaseImageOperator
from .processors import Processor, DatasetProcessor
from ..data_io import GDALDatasetIO


class SimilarityScoreOperator(BaseImageOperator):

    def __init__(
        self,
        allow_resize: bool = True,
        compare_nonzero: bool = True,
        tm_metric=cv2.TM_CCOEFF_NORMED,
        log_keys: list[str] = [],
        acceptance_threshold: float = 0.99,
    ) -> None:
        self.allow_resize = allow_resize
        self.compare_nonzero = compare_nonzero
        self.tm_metric = tm_metric
        self.log_keys = log_keys
        self.acceptance_threshold = acceptance_threshold

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:

        if src_img.shape[-1] != dst_img.shape[-1]:
            if dst_img.shape[-1] < src_img.shape[-1]:
                raise ValueError(
                    'Destination image must have as many or more channels '
                    'than source image.'
                )
            dst_img = dst_img[..., :src_img.shape[-1]]

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

    def assert_equal(self, src_img: np.ndarray, dst_img: np.ndarray) -> None:
        results = self.operate(src_img, dst_img)
        assert results['score'] > self.acceptance_threshold, (
            f'Images have a score of {results["score"]}'
        )


class DatasetScorer(DatasetProcessor):

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:

        # Combine the images
        # TODO: image_operator is more-general,
        #       but image_blender is more descriptive
        results = self.image_operator.operate(
            src['image'],
            dst['image'],
        )
        self.update_log(self.image_operator.log)

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ):

        if results['return_code'] == 'success':
            row['score'] = results['score']
        else:
            row['score'] = np.nan

        row['return_code'] = results['return_code']

        return row


class ReferencedImageScorer(Processor):
    # TODO: Consistent naming: registered images or referenced images

    def __init__(self, crs: pyproj.CRS = None, *args, **kwargs):

        self.crs = crs

        super().__init__(*args, **kwargs)

    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:

        src_dataset = GDALDatasetIO.load(
            row['filepath'],
            crs=self.crs,
        )

        return {'dataset': src_dataset}

    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:

        if pd.isna(row['output_filepath']):
            return {'dataset': None}

        dst_dataset = GDALDatasetIO.load(
            row['output_filepath'],
            crs=self.crs,
        )

        return {'dataset': dst_dataset}

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:

        if dst['dataset'] is None:
            return {}

        src_x_bounds, src_y_bounds, src_pixel_width, src_pixel_height = \
            GDALDatasetIO.get_bounds_from_dataset(src['dataset'])
        dst_x_bounds, dst_y_bounds, dst_pixel_width, dst_pixel_height = \
            GDALDatasetIO.get_bounds_from_dataset(dst['dataset'])

        results = {
            'x_size_diff': (
                dst['dataset'].RasterXSize - src['dataset'].RasterXSize),
            'y_size_diff': (
                dst['dataset'].RasterYSize - src['dataset'].RasterYSize),
            'x_min_diff': dst_x_bounds[0] - src_x_bounds[0],
            'x_max_diff': dst_x_bounds[1] - src_x_bounds[1],
            'y_min_diff': dst_y_bounds[0] - src_y_bounds[0],
            'y_max_diff': dst_y_bounds[1] - src_y_bounds[1],
            'pixel_width_diff': dst_pixel_width - src_pixel_width,
            'pixel_height_diff': dst_pixel_height - src_pixel_height,
        }
        results['center_diff'] = 0.5 * np.sqrt(
            (results['x_min_diff'] + results['x_max_diff'])**2.
            + (results['y_min_diff'] + results['y_max_diff'])**2.
        )

        # If we actually want to compare the image values
        if self.image_operator is not None:
            results['image_operator_score'] = self.image_operator.operate(
                src['dataset'].ReadAsArray(),
                dst['dataset'].ReadAsArray(),
            )

        results['score'] = results['center_diff']

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ) -> pd.Series:

        # Combine
        for key, item in results.items():
            row[key] = item

        return row
