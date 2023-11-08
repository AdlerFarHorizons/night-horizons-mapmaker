'''Test file for pipelines.
'''

import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import night_horizons.utils as utils
import night_horizons.pipelines as pipelines


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
        fps = utils.discover_data(image_dir)

        pipeline = self.pipelines.referenced_mosaic(self.fp)
        mosaic = pipeline.fit_transform(fps)

        # Compare to one of the input images
        i = 0
        fp = fps.iloc[i]
        row = pipeline.named_steps['geobounds'].transform([fp,])
        actual_img = utils.load_image(fp)
        score = mosaic.score(
            actual_img,
            row['x_min'],
            row['x_max'],
            row['y_min'],
            row['y_max']
        )
        assert score > 


