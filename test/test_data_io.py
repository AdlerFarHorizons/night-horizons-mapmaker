import unittest

import numpy as np
import pandas as pd
import scipy
import pyproj
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.data_io import GDALDatasetIO


class TestGDALDatasetIO(unittest.TestCase):

    def test_convert(self):

        fp = './test/test_data/referenced_images/Geo 836109848_1.tif'
        new_crs = pyproj.CRS('EPSG:3857')

        io = GDALDatasetIO()
        dataset = io.load(fp)

        # The conversion is meaningless if the CRS is the same
        assert pyproj.CRS(dataset.GetProjection()) != new_crs

        dataset2 = io.convert(dataset, new_crs)
        (
            x_bounds, y_bounds,
            pixel_width, pixel_height,
        ) = io.get_bounds_from_dataset(dataset2)
        np.testing.assert_allclose(
            x_bounds[1] - x_bounds[0],
            pixel_width * dataset2.RasterXSize,
        )
        np.testing.assert_allclose(
            y_bounds[1] - y_bounds[0],
            -pixel_height * dataset2.RasterYSize,
        )
