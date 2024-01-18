'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd

from night_horizons.container import DIContainer
from night_horizons.data_io import GDALDatasetIO, RegisteredImageIO
from night_horizons.io_manager import IOManager
import night_horizons.transformers.raster as raster


class TestRasterCoordinateTransformer(unittest.TestCase):

    def setUp(self):

        x_bounds = np.array([-9599524.7998918, -9590579.50992268])
        y_bounds = np.array([4856260.998546081, 4862299.303607852])
        self.X = pd.Series({
            'x_min': x_bounds[0],
            'x_max': x_bounds[1],
            'y_min': y_bounds[0],
            'y_max': y_bounds[1],
        })
        self.X = pd.DataFrame([self.X])

        container = DIContainer('./test/test_transformers/config.yml')
        container.register_service('io_manager', IOManager)

        # Convenient access to container
        self.container = container
        self.settings = container.config
        self.io_manager = self.container.get_service('io_manager')

    def test_consistent(self):

        dataset = GDALDatasetIO.load(
            self.io_manager.input_filepaths['referenced_images'][0]
        )

        transformer = raster.RasterCoordinateTransformer()
        transformer.fit_to_dataset(dataset)

        X_t = transformer.transform(self.X)
        X_reversed = transformer.transform(X_t, direction='pixel_to_physical')

        pd.testing.assert_frame_equal(X, X_reversed)
