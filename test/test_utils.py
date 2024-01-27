'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd

import night_horizons.utils as utils


class TestLoadImage(unittest.TestCase):

    def test_tiff(self):

        fp = '/data/test_data/referenced_images/Geo 843083290_1.tif'
        img = utils.load_image(fp)

        assert len(img.shape) == 3

    def test_raw(self):

        fp = ('/data/test_data/images/23085686/'
              '20220413_221313_1020286912_0_50_3.raw')
        img = utils.load_image(fp)

        assert len(img.shape) == 3

        # Max should be close to 255
        assert img.max() > 250


class TestDiscoverData(unittest.TestCase):

    def test_discover_data(self):

        image_dir = '/data/test_data/images'
        expected_fps = [
            ('/data/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('/data/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = utils.discover_data(image_dir)
        assert list(fps) == expected_fps

    def test_discover_data_exts(self):

        image_dir = '/data/test_data'
        expected_fps_a = [
            ('/data/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('/data/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = utils.discover_data(image_dir, ['raw', 'tif', 'tif'])

        actual_fps_a = [_ for _ in list(fps) if 'test_data/images' in _]
        assert list(actual_fps_a) == expected_fps_a

        actual_fps_b = [_ for _ in list(fps) if 'referenced_images' in _ ]
        assert len(actual_fps_b) > 0

    def test_discover_data_pattern(self):

        image_dir = '/data/test_data'
        expected_fps = [
            '/data/test_data/referenced_images/Geo 843083290_1.tif',
            '/data/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = utils.discover_data(
            image_dir,
            extension=['.tif', '.tiff'],
            pattern=r'Geo\s\d+_\d.tif'
        )
        assert list(fps) == expected_fps


class TestStoreParameters(unittest.TestCase):

    def test_functional(self):

        class MyClass:

            @utils.store_parameters
            def __init__(self, a, b, c=5, dog=True):
                pass

        a = 'sentence'
        b = 'another'
        c = 3
        my_class = MyClass(a, b, c) 

        assert my_class.a == 'sentence'
        assert my_class.b == b
        assert my_class.c == 3
