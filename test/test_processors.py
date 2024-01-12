import os
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.container import DIContainer
from night_horizons.io_manager import IOManager
from night_horizons.raster import ReferencedImage
from night_horizons.image_processing.processors import DatasetRegistrar
from night_horizons.image_processing.operators import BaseImageOperator
from night_horizons.image_processing.scorers import SimilarityScorer


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
        container = DIContainer(
            './test/config.yml', local_options=local_options)
        dataio_services = container.register_dataio_services()

        # Register specific services
        def register_io_manager(*args, **kwargs):
            data_ios = {
                name: container.get_service(name) for name in dataio_services
            }
            return IOManager(data_ios=data_ios, *args, **kwargs)
        container.register_service('io_manager', register_io_manager)
        container.register_service(
            'image_operator', MagicMock(spec=BaseImageOperator))
        container.register_service('image_scorer', SimilarityScorer)

        # Register the DatasetRegistrar
        container.register_service(
            'dataset_registrar',
            lambda use_safe_process=False, *args, **kwargs:
                DatasetRegistrar(
                    io_manager=container.get_service('io_manager'),
                    image_operator=container.get_service('image_operator'),
                    use_safe_process=use_safe_process,
                    *args, **kwargs
                )
        )

        self.container = container
        self.settings = self.container.config

    def compare_referenced_images(self, expected_fp, actual_fp):

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
        score = image_scorer.operate(actual_image, expected_image)
        assert score > 0.9, f'Image has a score of {score}'


class TestDatasetRegistrar(TestProcessorBase):

    def test_save_image_as_dataset(self):

        processor = self.container.get_service('dataset_registrar')

        expected_fp = (
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif'
        )
        original_image = ReferencedImage.open(expected_fp)

        row = pd.Series({
            'x_min': original_image.cart_bounds[0][0],
            'x_max': original_image.cart_bounds[0][1],
            'y_min': original_image.cart_bounds[1][0],
            'y_max': original_image.cart_bounds[1][1],
        })
        row.name = 0

        processor.save_image_as_new_dataset(
            img=original_image.img_int,
            row=row,
            x_off_new=0,
            y_off_new=0,
        )

        self.compare_referenced_images(
            expected_fp=expected_fp,
            actual_fp='./test/test_data/temp/referenced_images/img_000000.tiff'
        )

    def test_consistency(self):

        # Load padded image
        # (small referenced mosaic that is a single padded image)

        # Match to padded image

        assert False
