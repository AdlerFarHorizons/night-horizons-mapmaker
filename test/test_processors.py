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

from night_horizons.image_processing import processors, scorers
from night_horizons.pipeline import create_mapmaker
from night_horizons.raster import Image, ReferencedImage
from night_horizons.transformers.raster import RasterCoordinateTransformer


class TestDatasetRegistrar(unittest.TestCase):

    def setUp(self):

        # Register services
        local_options = {
            'mapmaker': {'map_type': 'sequential'},
            'io_manager': {
                'output_dir': '/data/night_horizons_test_data/temp',
                'output_description': {
                    'referenced_images':
                        'referenced_images/img_ind{:06d}.tiff',
                },
            },
        }

        # Create container
        mapmaker = create_mapmaker(
            './test/config.yml',
            local_options=local_options,
        )

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
        # Also the scorer we use
        mapmaker.container.register_service(
            'image_scorer',
            scorers.SimilarityScoreOperator
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
        pixel_diff_threshold=1,
    ):
        """Compare two referenced images.
        TODO: I pulled this out of the functions and made more general, but
        it's not used elsewhere...

        Args:
            expected_fp (str): File path of the expected image.
            actual_fp (str): File path of the actual image.
            acceptance_threshold (float, optional): Threshold for accepting
                the image similarity score. Defaults to 0.99.
        """

        assert os.path.isfile(actual_fp), f'File {actual_fp} not found.'
        actual_image = ReferencedImage.open(
            actual_fp,
            cart_crs=self.container.get_service('crs'),
        )

        expected_image = ReferencedImage.open(
            expected_fp,
            cart_crs=self.container.get_service('crs'),
        )

        # Compare image shape
        np.testing.assert_allclose(
            actual_image.img_shape,
            expected_image.img_shape,
            atol=pixel_diff_threshold,
        )

        # Compare image bounds
        np.testing.assert_allclose(
            actual_image.cart_bounds,
            expected_image.cart_bounds,
            pixel_diff_threshold * actual_image.get_pixel_widths()[0],
        )

        # Compare image contents
        image_scorer = self.container.get_service('image_scorer')
        image_scorer.assert_equal(actual_image.img_int, expected_image.img_int)

    def test_store_results(self):

        processor = self.container.get_service('dataset_registrar')

        # Dataset
        expected_fp = (
            '/data/night_horizons_test_data/referenced_images/Geo 225856_1473511261_0.tif'
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
            actual_fp=(
                '/data/night_horizons_test_data/temp/referenced_images/img_ind000000.tiff'
            ),
        )

    def test_end_to_end(self):
        '''This test loads a registered image, pads it, and then checks that
        we can find it again using dataset registrar. But first it overwrites
        the registered image with a simpler one
        '''

        original_fp = (
            '/data/night_horizons_test_data/referenced_images/Geo 225856_1473511261_0.tif'
        )
        original_image = ReferencedImage.open(original_fp)

        # Overwrite the image with a simpler one:
        # An example image with a white frame around the border of the
        # original image
        original_image.img_int = np.zeros(
            original_image.img_int.shape,
            dtype=original_image.img_int.dtype,
        )
        # White frame
        original_image.img_int[:50, :, :3] = 255
        original_image.img_int[-50:, :, :3] = 255
        original_image.img_int[:, :50, :3] = 255
        original_image.img_int[:, -50:, :3] = 255
        # Example image
        example_image = Image.open(
            '/data/night_horizons_test_data/feature_matching/tree_4.1.06.tiff',
            dtype=original_image.img_int.dtype,
        )
        original_image.img_int[
            50:50 + example_image.img_shape[0],
            50:50 + example_image.img_shape[0],
        ] = example_image.img_int

        # Save the image
        expected_fp = '/data/night_horizons_test_data/temp/source/img_000000.tiff'
        if os.path.isfile(expected_fp):
            os.remove(expected_fp)
        original_image.save(expected_fp)

        self.end_to_end_test(expected_fp)

    def test_realistic_end_to_end(self):
        '''This test loads a registered image, pads it, and then checks that
        we can find it again using dataset registrar.
        It does this twice.
        '''

        test_dir = '/data/night_horizons_test_data/referenced_images'
        filenames = [
            # 'Geo 225856_1473511261_0.tif',
            'Geo 836109848_1.tif',
        ]

        # Lower threshold, since realistic images are more complex
        self.container.config.setdefault(
            'image_scorer',
            {},
        )['acceptance_threshold'] = 0.9

        for filename in filenames:
            expected_fp = os.path.join(test_dir, filename)
            self.end_to_end_test(expected_fp)

    def test_sequential_end_to_end(self):
        '''This test loads a registered image, pads it, and then matches
        an overlapping registered image to it and checks the results.
        This is the main logic of the sequential mosaic maker.
        '''

        test_dir = '/data/night_horizons_test_data/referenced_images'
        original_fp = os.path.join(test_dir, 'Geo 836109848_1.tif')
        expected_fp = os.path.join(test_dir, 'Geo 843083290_1.tif')

        # Lower threshold, since realistic images are more complex
        self.container.config.setdefault(
            'image_scorer',
            {},
        )['acceptance_threshold'] = 0.5

        self.end_to_end_test(
            expected_fp=expected_fp,
            original_fp=original_fp,
            padding=500,
            pixel_diff_threshold=80,
        )

    def end_to_end_test(
        self,
        expected_fp: str,
        original_fp: str = None,
        padding: int = 100,
        *args, **kwargs
    ):

        if original_fp is None:
            original_fp = expected_fp

        original_image = ReferencedImage.open(original_fp)

        processor: processors.DatasetRegistrar = \
            self.container.get_service('dataset_registrar')

        # Revised version that's padded
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
            options=[],
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
            'x_min': x_bounds_padded[0],
            'x_max': x_bounds_padded[1],
            'y_min': y_bounds_padded[0],
            'y_max': y_bounds_padded[1],
            'x_off': 0,
            'y_off': 0,
            'x_size': containing_img.shape[1],
            'y_size': containing_img.shape[0],
        })
        row.name = 0

        # Match to padded image
        row = processor.process_row(
            i=0,
            row=row,
            resources=resources,
        )

        self.compare_referenced_images(
            expected_fp=expected_fp,
            actual_fp=(
                '/data/night_horizons_test_data/temp/referenced_images/img_ind000000.tiff'
            ),
            *args, **kwargs
        )

