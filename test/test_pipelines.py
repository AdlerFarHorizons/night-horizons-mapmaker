'''Test file for pipelines.
'''

import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import night_horizons_mapmaker.preprocess as preprocess
import night_horizons_mapmaker.pipelines as pipelines


class TestGeoreference(unittest.TestCase):

    def setUp(self):

        self.pipelines = pipelines.GeoreferencingPipelines()

    def test_sensor_georeferencing(self):

        image_dir = './test/test_data/referenced_images'
        fps = preprocess.discover_data(image_dir)

        sensor_ref_pipeline = self.pipelines.sensor_georeferencing()

        fps_train, fps_test = train_test_split(fps, test_size=0.2)

        sensor_ref_pipeline.fit(fps_train)
        y_pred = sensor_ref_pipeline.predict(fps_test)

        expected_cols = (
            preprocess.GEOTRANSFORM_COLS
            + ['estimated_spatial_error',]
        )
        assert y_pred.columns == expected_cols
