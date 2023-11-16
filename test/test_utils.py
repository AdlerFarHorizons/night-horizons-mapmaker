'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd

import night_horizons.utils as utils


class TestDiscoverData(unittest.TestCase):

    def test_discover_data(self):

        image_dir = './test/test_data/images'
        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = utils.discover_data(image_dir)
        assert list(fps) == expected_fps

    def test_discover_data_exts(self):

        image_dir = './test/test_data'
        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif',
            './test/test_data/referenced_images/Geo 843083290_1.tif',
            './test/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = utils.discover_data(image_dir, ['raw', 'tif', 'tiff'])
        assert list(fps) == expected_fps

    def test_discover_data_pattern(self):

        image_dir = './test/test_data'
        expected_fps = [
            './test/test_data/referenced_images/Geo 843083290_1.tif',
            './test/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = utils.discover_data(
            image_dir,
            extension=['.tif', '.tiff'],
            pattern=r'Geo\s\d+_\d.tif'
        )
        assert list(fps) == expected_fps
