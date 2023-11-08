'''Test file for pipelines.
'''

import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import night_horizons.utils as utils
import night_horizons.pipelines as pipelines
import night_horizons.metrics as metrics


class TestMosaic(unittest.TestCase):

    def setUp(self):

        self.pipelines = pipelines.MosaicPipelines()
        self.fp = './test/test_data/mosaics/temp.tiff'

        os.makedirs(os.path.dirname(self.fp), exist_ok=True)
        if os.path.isfile(self.fp):
            os.remove(self.fp)

    def tearDown(self):

        if os.path.isfile(self.fp):
            os.remove(self.fp)

    def test_referenced_mosaic(self):

        image_dir = './test/test_data/referenced_images'
        fps = utils.discover_data(image_dir, extension=['tif', 'tiff'])

        pipeline = self.pipelines.referenced_mosaic(self.fp)
        mosaic = pipeline.fit_transform(fps)

        # Check the score
        score = pipeline.score(fps)
        assert score > metrics.R_ACCEPT


