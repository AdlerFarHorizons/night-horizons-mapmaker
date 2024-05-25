'''Test file for preprocessing.
'''

import os
import unittest
import difflib
import pprint

import numpy as np
import pandas as pd

import night_horizons.utils as utils


def assert_sorted_lists_equal(list1, list2):
    assert sorted(list1) == sorted(list2), (
        "Lists differ:\n" + '\n'.join(difflib.ndiff(
            pprint.pformat(list1).splitlines(),
            pprint.pformat(list2).splitlines())
        )
    )


class TestDiscoverData(unittest.TestCase):

    def setUp(self):

        self.expected_fps_raw = [
            ('/data/input/nitelite.images/220513-FH135/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('/data/input/nitelite.images/220513-FH135/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
            ('/data/input/nitelite.images/220513-FH135/23085687/'
             'Geo 836109848_1.tif'),
            ('/data/input/nitelite.images/240203-FH145/23085687/'
             '10_1707005484.653204_23085687_1_img_0.raw'),
            ('/data/input/nitelite.images/240203-FH145/23085687/'
             '10_1707005484.683574_23085687_1_img_1.raw'),
            ('/data/input/nitelite.images/240203-FH145/23085687/'
             '10_1707005484.714646_23085687_1_img_2.raw'),
            ('/data/input/nitelite.images/240203-FH145/23085687/'
             '737_1707014395.8863537_23085687_1_img.tiff'),
        ]

    def test_discover_data(self):

        image_dir = '/data/input/nitelite.images'

        fps = utils.discover_data(image_dir)
        assert list(fps) == self.expected_fps_raw

    def test_discover_data_exts(self):

        image_dir = '/data/'

        fps = utils.discover_data(image_dir, extension=['raw', 'tif', 'tiff'])

        actual_fps_a = [_ for _ in list(fps) if '/nitelite.images/' in _]
        assert list(actual_fps_a) == self.expected_fps_raw

        actual_fps_b = [_ for _ in list(fps) if 'nitelite.referenced-images' in _]
        assert len(actual_fps_b) > 0

    def test_discover_data_pattern(self):

        image_dir = '/data/input/nitelite.referenced-images'
        expected_fps = [
            '/data/input/nitelite.referenced-images/220513-FH135/Geo 843083290_1.tif',
            '/data/input/nitelite.referenced-images/220513-FH135/Geo 836109848_1.tif',
        ]

        fps = utils.discover_data(
            image_dir,
            extension=['.tif', '.tiff'],
            pattern=r'Geo\s\d+_\d.tif'
        )
        assert_sorted_lists_equal(list(fps), expected_fps)
