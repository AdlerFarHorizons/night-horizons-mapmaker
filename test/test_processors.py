import os
import shutil
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.image_processing import processors
from night_horizons.mapmake import SequentialMosaicMaker
from night_horizons.raster import ReferencedImage
from night_horizons.transformers.raster import RasterCoordinateTransformer


class TestProcessorBase(unittest.TestCase):

    def setUp(self):

        # Register services
        local_options = {
            'io_manager': {
                'output_dir': './test/test_data/temp',
                'output_description': {
                    'referenced_images': 'referenced_images/img_{:06d}.tiff',
                },
            },
        }

        # Create container
        mapmaker = SequentialMosaicMaker(
            './test/config.yml', local_options=local_options)

        # Register the DatasetRegistrar
        mapmaker.container.register_service(
            'dataset_registrar',
            lambda use_safe_process=False, *args, **kwargs:
                processors.DatasetRegistrar(
                    io_manager=mapmaker.container.get_service('io_manager'),
                    image_operator=mapmaker.container.get_service(
                        'image_operator'),
                    use_safe_process=use_safe_process,
                    *args, **kwargs
                )
        )
        self.mapmaker = mapmaker

        # Convenient access to container
        self.container = self.mapmaker.container
        self.settings = self.container.config
        self.io_manager = self.container.get_service('io_manager')

    def tearDown(self):
        if os.path.isdir(self.io_manager.output_dir):
            shutil.rmtree(self.io_manager.output_dir)

    def compare_referenced_images(
        self,
        expected_fp,
        actual_fp,
        acceptance_threshold=0.99,
    ):
        """Compare two referenced images.

        Args:
            expected_fp (str): File path of the expected image.
            actual_fp (str): File path of the actual image.
            acceptance_threshold (float, optional): Threshold for accepting
                the image similarity score. Defaults to 0.99.
        """

        assert os.path.isfile(actual_fp), f'File {actual_fp} not found.'
        actual_image = ReferencedImage.open(
            actual_fp,
            cart_crs_code=self.settings['global']['crs'],
        )

        expected_image = ReferencedImage.open(
            expected_fp,
            cart_crs_code=self.settings['global']['crs'],
        )

        # Compare image shape
        np.testing.assert_allclose(
            actual_image.img_shape,
            expected_image.img_shape,
        )

        # Compare image bounds
        np.testing.assert_allclose(
            actual_image.cart_bounds,
            expected_image.cart_bounds,
        )

        # Compare image contents
        image_scorer = self.container.get_service('image_scorer')
        score_results = image_scorer.operate(
            actual_image.img_int, expected_image.img_int)
        score = score_results['score']
        assert score > acceptance_threshold, f'Image has a score of {score}'


class TestDatasetRegistrar(TestProcessorBase):

    def test_store_results(self):

        processor = self.container.get_service('dataset_registrar')

        # Dataset
        expected_fp = (
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif'
        )
        original_image = ReferencedImage.open(expected_fp)
        transformer = RasterCoordinateTransformer()
        transformer.fit_to_dataset(original_image.dataset)

        # X data
        row = pd.Series({
            'x_min': original_image.cart_bounds[0][0],
            'x_max': original_image.cart_bounds[0][1],
            'y_min': original_image.cart_bounds[1][0],
            'y_max': original_image.cart_bounds[1][1],
            'x_off': 0,
            'y_off': 0,
            'x_size': original_image.img_shape[1],
            'y_size': original_image.img_shape[0],
        })
        row.name = 0

        processor.store_results(
            i=0,
            row=row,
            resources={
                'dataset': original_image.dataset,
                'transformer': transformer,
            },
            results={
                'blended_image': original_image.img_int,
                'warped_image': original_image.img_int,
                'warped_bounds': (0, 0, row['x_size'], row['y_size']),
                'return_code': 'success',
            },
        )

        self.compare_referenced_images(
            expected_fp=expected_fp,
            actual_fp='./test/test_data/temp/referenced_images/img_000000.tiff'
        )

    def test_end_to_end(self):
        '''This test loads a registered image, pads it, and then checks that
        we can find it again using dataset registrar. But first it overwrites
        the registered image with a simpler one
        '''

        expected_fp = (
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif'
        )
        original_image = ReferencedImage.open(expected_fp)

        # Overwrite the image with a simpler one
        original_image.img_int = np.zeros(
            original_image.img_int.shape,
            dtype=original_image.img_int.dtype,
        )
        original_image.img_int[:50, :, :3] = 255
        original_image.img_int[50:, :, :3] = 255
        original_image.img_int[:, :50, :3] = 255
        original_image.img_int[:, 50:, :3] = 255
        original_image.img = original_image.img_int / 255

        self.end_to_end_test(expected_fp, original_image)

    def test_realistic_end_to_end(self):
        '''This test loads a registered image, pads it, and then checks that
        we can find it again using dataset registrar.
        '''

        expected_fp = (
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif'
        )
        original_image = ReferencedImage.open(expected_fp)

        self.end_to_end_test(expected_fp, original_image)

    def end_to_end_test(
        self,
        expected_fp: str,
        original_image: ReferencedImage,
    ):

        processor = self.container.get_service('dataset_registrar')

        # Revised version that's padded
        padding = 100
        containing_img = np.zeros(
            (original_image.img_shape[0] + 2 * padding,
             original_image.img_shape[1] + 2 * padding,
             original_image.img_int.shape[2]),
            dtype=original_image.img_int.dtype,
        )
        containing_img[
            padding:-padding,
            padding:-padding,
            :
        ] = original_image.img_int
        dx, dy = original_image.get_pixel_widths()
        x_bounds_padded = (
            original_image.cart_bounds[0][0] - padding * dx,
            original_image.cart_bounds[0][1] + padding * dx,
        )
        y_bounds_padded = (
            original_image.cart_bounds[1][0] - padding * dy,
            original_image.cart_bounds[1][1] + padding * dy,
        )
        dataset = original_image.io.save(
            filepath='',
            img=containing_img,
            x_bounds=x_bounds_padded,
            y_bounds=y_bounds_padded,
            crs=original_image.cart_crs,
            driver='MEM',
        )
        transformer = RasterCoordinateTransformer()
        transformer.fit_to_dataset(dataset)
        resources = {
            'dataset': dataset,
            'transformer': transformer,
        }

        # Row containing pre-processing information
        row = pd.Series({
            'filepath': expected_fp,
            'x_min': original_image.cart_bounds[0][0],
            'x_max': original_image.cart_bounds[0][1],
            'y_min': original_image.cart_bounds[1][0],
            'y_max': original_image.cart_bounds[1][1],
            'x_off': 0,
            'y_off': 0,
            'x_size': original_image.img_shape[1],
            'y_size': original_image.img_shape[0],
        })
        row.name = 0

        # Fit properties
        processor.x_size_ = dataset.RasterXSize
        processor.y_size_ = dataset.RasterYSize

        # Match to padded image
        row = processor.process_row(
            i=0,
            row=row,
            resources=resources,
        )

        self.compare_referenced_images(
            expected_fp=expected_fp,
            actual_fp='./test/test_data/temp/referenced_images/img_000000.tiff'
        )
