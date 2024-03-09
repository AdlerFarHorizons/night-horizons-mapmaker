'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import pyproj

from night_horizons.container import DIContainer
from night_horizons.data_io import GDALDatasetIO, RegisteredImageIO
from night_horizons.io_manager import IOManager
import night_horizons.transformers.raster as raster
from night_horizons.pipeline import create_stage


class TestRasterCoordinateTransformer(unittest.TestCase):

    def setUp(self):

        self.random_state = check_random_state(42)

        self.mapmaker = create_stage('./test/test_transformers/config.yml')
        self.io_manager = self.mapmaker.container.get_service('io_manager')

        # Load the example data we'll use
        self.dataset = GDALDatasetIO.load(
            self.io_manager.input_filepaths['referenced_images'][0],
            crs=self.mapmaker.container.get_service('crs'),
        )
        (
            self.x_bounds,
            self.y_bounds,
            self.pixel_width,
            self.pixel_height,
        ) = GDALDatasetIO.get_bounds_from_dataset(
            self.dataset,
        )

    def test_to_pixel(self):

        # Make a dataframe that is a single entry--the full size of the image
        X = pd.Series({
            'x_min': self.x_bounds[0],
            'x_max': self.x_bounds[1],
            'y_min': self.y_bounds[0],
            'y_max': self.y_bounds[1],
        })
        X = pd.DataFrame([X])

        # Fit the transformer
        transformer = raster.RasterCoordinateTransformer()
        transformer.fit_to_dataset(self.dataset)

        # Test that the transformer works
        X_t = transformer.transform(X.copy())

        X_expected = pd.Series({
            'x_off': 0,
            'y_off': 0,
            'x_size': self.dataset.RasterXSize,
            'y_size': self.dataset.RasterYSize,
        })
        X_expected = pd.DataFrame([X_expected])
        pd.testing.assert_frame_equal(X_expected, X_t[X_expected.columns])

    def test_to_physical(self):

        # Make a dataframe that is a single entry--the full size of the image
        X = pd.Series({
            'x_off': 0,
            'y_off': 0,
            'x_size': self.dataset.RasterXSize,
            'y_size': self.dataset.RasterYSize,
        })
        X = pd.DataFrame([X])

        # Fit the transformer
        transformer = raster.RasterCoordinateTransformer()
        transformer.fit_to_dataset(self.dataset)

        # Test that the transformer works
        X_t = transformer.transform(X.copy(), direction='to_physical')

        X_expected = pd.Series({
            'x_min': self.x_bounds[0],
            'x_max': self.x_bounds[1],
            'y_min': self.y_bounds[0],
            'y_max': self.y_bounds[1],
        })
        X_expected = pd.DataFrame([X_expected])
        pd.testing.assert_frame_equal(X_expected, X_t[X_expected.columns])

    def test_consistent(self):

        # Create the test data
        X = pd.DataFrame({
            'x_min': self.random_state.uniform(
                self.x_bounds[0], self.x_bounds[1], 100),
            'x_max': self.random_state.uniform(
                self.x_bounds[0], self.x_bounds[1], 100),
            'y_min': self.random_state.uniform(
                self.y_bounds[0], self.y_bounds[1], 100),
            'y_max': self.random_state.uniform(
                self.y_bounds[0], self.y_bounds[1], 100),
        })
        # Drop situations where max < min
        X = X.drop(X[X['x_max'] < X['x_min']].index)
        X = X.drop(X[X['y_max'] < X['y_min']].index)
        X['padding'] = 0.
        X['pixel_width'] = self.pixel_width
        X['pixel_height'] = self.pixel_height

        # Fit the transformer
        transformer = raster.RasterCoordinateTransformer()
        transformer.fit(X)

        # Test that the transformer works
        X_t = transformer.transform(X.copy())
        X_reversed = transformer.transform(X_t.copy(), direction='to_physical')
        pd.testing.assert_frame_equal(X, X_reversed[X.columns])

        # Check that a failure does indeed occur
        transformer.pixel_height_ *= 2
        X_reversed = transformer.transform(X_t.copy(), direction='to_physical')
        with self.assertRaises(AssertionError) as context:
            pd.testing.assert_frame_equal(X, X_reversed[X.columns])

    def test_consistent_fit_params(self):

        # Create the test data
        n = 1000
        X = pd.DataFrame({
            'x_min': self.random_state.uniform(
                self.x_bounds[0], self.x_bounds[1], n),
            'x_max': self.random_state.uniform(
                self.x_bounds[0], self.x_bounds[1], n),
            'y_min': self.random_state.uniform(
                self.y_bounds[0], self.y_bounds[1], n),
            'y_max': self.random_state.uniform(
                self.y_bounds[0], self.y_bounds[1], n),
        })
        # Drop situations where max < min
        X = X.drop(X[X['x_max'] < X['x_min']].index)
        X = X.drop(X[X['y_max'] < X['y_min']].index)
        X['padding'] = 0.
        X['pixel_width'] = self.pixel_width
        X['pixel_height'] = self.pixel_height
        dx_between_avg = (self.x_bounds[1] - self.x_bounds[0]) / len(X)
        dy_between_avg = (self.y_bounds[1] - self.y_bounds[0]) / len(X)
        d_between_avg = np.sqrt(dx_between_avg**2 + dy_between_avg**2)

        # Fit the transformer
        transformer = raster.RasterCoordinateTransformer()
        transformer.fit(X)

        # Fit the transformer
        transformer2 = raster.RasterCoordinateTransformer()
        transformer2.fit_to_dataset(self.dataset)

        attrs_to_check = [
            'x_min_', 'x_max_',
            'y_min_', 'y_max_',
            'pixel_width_', 'pixel_height_',
            'x_size_', 'y_size_',
        ]
        for attr in attrs_to_check:
            # This checks that the fit parameters are equal, assuming the
            # random sample fills out the full range of the dataset
            # (+- a few times the average distance between points).
            # This can still fail for an unlucky seed.
            np.testing.assert_allclose(
                getattr(transformer, attr),
                getattr(transformer2, attr),
                atol=2 * d_between_avg
            )
