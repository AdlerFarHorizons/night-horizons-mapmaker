'''Test file for pipelines.
'''

import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import night_horizons.utils as utils
import night_horizons.raster as raster
import night_horizons.pipelines as pipelines
import night_horizons.metrics as metrics


class BaseTester(unittest.TestCase):

    def setUp(self):

        self.mosaic_fp = './test/test_data/mosaics/temp.tiff'
        os.makedirs(os.path.dirname(self.mosaic_fp), exist_ok=True)
        if os.path.isfile(self.mosaic_fp):
            os.remove(self.mosaic_fp)

        image_dir = './test/test_data/referenced_images'
        self.fps = utils.discover_data(image_dir, extension=['tif', 'tiff'])

    def tearDown(self):

        if os.path.isfile(self.mosaic_fp):
            os.remove(self.mosaic_fp)


class TestSensorGeoreference(BaseTester):

    def setUp(self):

        super().setUp()

        self.pipeline, self.y_pipeline = \
            pipelines.GeoreferencePipelines.sensor_georeference()

        # Accurate to within 2 km
        self.acceptance_criteria = 2000.

    def test_score(self):
        '''For this test we're scoring the values it was trained on,
        so this is not a rigorous test.
        '''

        # Fit
        y = self.y_pipeline.fit_transform(self.fps)
        self.pipeline.fit(
            self.fps,
            y,
            nitelite__img_log_fp='test/test_data/metadata/image.log',
            nitelite__imu_log_fp='test/test_data/metadata/PresIMULog.csv',
            nitelite__gps_log_fp='test/test_data/metadata/GPSLog.csv',
        )

        # Score
        score = self.pipeline.score(self.fps, y)
        assert score < self.acceptance_criteria


class TestReferencedMosaic(BaseTester):

    def setUp(self):

        super().setUp()

        self.pipeline = pipelines.MosaicPipelines.referenced_mosaic(
            self.mosaic_fp
        )

    def test_score(self):

        X_transformed = self.pipeline.fit_transform(self.fps)

        # Check the score
        score = self.pipeline.score(self.fps)
        assert score > metrics.R_ACCEPT

    def test_bounds_to_offset(self):

        X_transformed = self.pipeline.fit_transform(self.fps)

        # Bounds for the whole dataset
        reffed_mosaic = self.pipeline.named_steps['mosaic']
        (
            x_offset,
            y_offset,
            x_count,
            y_count,
        ) = reffed_mosaic.bounds_to_offset(
            reffed_mosaic.x_min_, reffed_mosaic.x_max_,
            reffed_mosaic.y_min_, reffed_mosaic.y_max_,
        )

        assert x_offset == 0
        assert y_offset == 0
        assert x_count == reffed_mosaic.dataset_.RasterXSize
        assert y_count == reffed_mosaic.dataset_.RasterYSize


class TestLessReferencedMosaic(BaseTester):

    def setUp(self):

        super().setUp()

        self.pipeline = pipelines.MosaicPipelines.less_referenced_mosaic(
            self.mosaic_fp
        )

    def test_score(self):

        X_transformed = self.pipeline.fit_transform(self.fps)

        # Check the score
        score = self.pipeline.score(self.fps)
        assert score > metrics.R_ACCEPT
