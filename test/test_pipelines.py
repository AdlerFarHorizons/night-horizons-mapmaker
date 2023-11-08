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


class TestReferencedMosaic(unittest.TestCase):

    def setUp(self):

        self.fp = './test/test_data/mosaics/temp.tiff'
        os.makedirs(os.path.dirname(self.fp), exist_ok=True)
        if os.path.isfile(self.fp):
            os.remove(self.fp)

        self.pipeline = pipelines.MosaicPipelines.referenced_mosaic(self.fp)

        image_dir = './test/test_data/referenced_images'
        self.fps = utils.discover_data(image_dir, extension=['tif', 'tiff'])

    def tearDown(self):

        if os.path.isfile(self.fp):
            os.remove(self.fp)

    def test_score(self):

        mosaic = self.pipeline.fit_transform(self.fps)

        # Check the score
        score = self.pipeline.score(self.fps)
        assert score > metrics.R_ACCEPT

    def test_external_consistency(self):

        mosaic = self.pipeline.fit_transform(self.fps)

        # Compare to one of the images
        i = 0
        fp = self.fps.iloc[i]
        reffed_image = raster.ReferencedImage.open(fp)
        x_bounds, y_bounds = reffed_image.cart_bounds
        actual_img = self.pipeline.named_steps['mosaic'].get_image(
            x_bounds[0],
            x_bounds[1],
            y_bounds[0],
            y_bounds[1],
        )

        # Check the score
        r = metrics.image_to_image_ccoeff(
            reffed_image.img_int,
            actual_img[:, :, :3]
        )
        assert r > metrics.R_ACCEPT
