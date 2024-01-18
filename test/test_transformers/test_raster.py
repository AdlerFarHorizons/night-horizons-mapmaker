'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from night_horizons.container import DIContainer
from night_horizons.data_io import GDALDatasetIO, RegisteredImageIO
from night_horizons.io_manager import IOManager
import night_horizons.transformers.raster as raster


class TestRasterCoordinateTransformer(unittest.TestCase):

    def setUp(self):

        self.random_state = check_random_state(42)

        x_bounds = np.array([-9599524.7998918, -9590579.50992268])
        y_bounds = np.array([4856260.998546081, 4862299.303607852])
        X = pd.Series({
            'x_min': x_bounds[0],
            'x_max': x_bounds[1],
            'y_min': y_bounds[0],
            'y_max': y_bounds[1],
        })
        X = pd.DataFrame([X])

        container = DIContainer('./test/test_transformers/config.yml')
        container.register_service('io_manager', IOManager)

        # Convenient access to container
        self.container = container
        self.settings = container.config
        self.io_manager = self.container.get_service('io_manager')

    def test_with_dataset_fit(self):

        # Load the example data for the fit
        dataset = GDALDatasetIO.load(
            self.io_manager.input_filepaths['referenced_images'][0]
        )
        (
            x_bounds,
            y_bounds,
            pixel_width,
            pixel_height,
            crs
        ) = GDALDatasetIO.get_bounds_from_dataset(dataset)

        # Create the test data
        X = pd.DataFrame({
            'x_min': self.random_state.uniform(x_bounds[0], x_bounds[1], 100),
            'x_max': self.random_state.uniform(x_bounds[0], x_bounds[1], 100),
            'y_min': self.random_state.uniform(y_bounds[0], y_bounds[1], 100),
            'y_max': self.random_state.uniform(y_bounds[0], y_bounds[1], 100),
        })
        # Fix situations where max < min
        X.loc[X['x_max'] > X['x_min'], 'x_max'] = \
            X['x_min'] - (X['x_max'] - X['x_min'])
        X.loc[X['y_max'] > X['y_min'], 'y_max'] = \
            X['y_min'] - (X['y_max'] - X['y_min'])

        # Fit the transformer
        transformer = raster.RasterCoordinateTransformer()
        transformer.fit_to_dataset(dataset)

        # Test that the transformer works
        X_t = transformer.transform(X.copy())
        X_reversed = transformer.transform(X_t.copy(), direction='to_physical')
        pd.testing.assert_frame_equal(X, X_reversed[X.columns])

        # Check that a failure does indeed occur
        transformer.pixel_height_ *= 2
        X_reversed = transformer.transform(X_t.copy(), direction='to_physical')
        with self.assertRaises(AssertionError) as context:
            pd.testing.assert_frame_equal(X, X_reversed[X.columns])
