'''Test file for preprocessing
'''

import os
import unittest

import numpy as np

import night_horizons_mapmaker.preprocess as preprocess


class TestDiscoverData(unittest.TestCase):

    def test_discover_data(self):

        image_dir = './test/test_data/images'
        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = preprocess.discover_data(image_dir)
        assert fps == expected_fps

    def test_discover_data_exts(self):

        image_dir = './test/test_data'
        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
            './test/test_data/referenced_images/Geo 225856_1473511261_0.tif',
            './test/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = preprocess.discover_data(image_dir, ['raw', 'tif*'])
        assert fps == expected_fps


class TestMetadataPreprocesser(unittest.TestCase):

    def setUp(self):

        self.transformer = preprocess.MetadataPreprocesser()
        image_dir = './test/test_data/images'
        self.fps = preprocess.discover_data(image_dir)

    def test_output(self):
        '''The output prior to any form of georeferencing.
        '''

        output_df = self.transformer.fit_transform(self.fps)
        expected_cols = ['filepaths', 'sensor_x', 'sensor_y']
        assert (~output_df.columns.isin(expected_cols)).sum() == 0
