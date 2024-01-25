import os
import unittest

import numpy as np
import pandas as pd
import scipy
import pyproj
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.data_io import GDALDatasetIO, RegisteredImageIO


class TestGDALDatasetIO(unittest.TestCase):

    def setUp(self):

        self.viirs_fp = (
            './test/test_data/other/'
            'VNP46A2.A2022353.h09v04.001.2022361121713.h5'
        )

        self.viirs_output_fp = self.viirs_fp.replace('.h5', '.tiff')
        if os.path.isfile(self.viirs_output_fp):
            os.remove(self.viirs_output_fp)

    def tearDown(self):
        if os.path.isfile(self.viirs_output_fp):
            os.remove(self.viirs_output_fp)

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

    def test_from_viirs_hdf5(self):

        expected_crs = pyproj.CRS('EPSG:4326')

        # Check basic loading
        io = GDALDatasetIO()
        dataset = io.load_from_viirs_hdf5(self.viirs_fp)
        assert dataset is not None
        assert pyproj.CRS(dataset.GetProjection()) == expected_crs
        dataset.FlushCache()
        dataset = None

        # Check the saved data loads
        dataset2 = io.load(self.viirs_output_fp)
        assert pyproj.CRS(dataset2.GetProjection()) == expected_crs

        # Check compatibility with RegisteredImageIO
        img, x_bounds, y_bounds = RegisteredImageIO.load(
            self.viirs_output_fp,
        )
        x_bounds = np.array(x_bounds)
        y_bounds = np.array(y_bounds)

        assert img.shape == (2400, 2400)
        assert ((-95 < x_bounds) & (x_bounds < -75)).all(), \
            'x_bounds doesnt look like Chicago'
        assert ((35 < y_bounds) & (y_bounds < 55)).all(), \
            'x_bounds doesnt look like Chicago'
