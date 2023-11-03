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

        # Image filetree info
        image_dir = './test/test_data/images'
        self.fps = preprocess.discover_data(image_dir)
        self.n_files = len(self.fps)

        # Metadata filetree info
        metadata_dir = './test/test_data/metadata'
        self.img_log_fp = os.path.join(metadata_dir, 'image.log')
        self.imu_log_fp = os.path.join(metadata_dir, 'PresIMULog.csv')
        self.gps_log_fp = os.path.join(metadata_dir, 'GPSLog.csv')

        # Preprocesser construction
        self.expected_cols = ['filepath', 'sensor_x', 'sensor_y']
        self.transformer = preprocess.MetadataPreprocesser(
            output_columns=self.expected_cols
        )

    def test_output(self):
        '''The output prior to any form of georeferencing.
        '''

        output_df = self.transformer.fit_transform(
            self.fps,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )
        assert len(output_df) == self.n_files
        assert (~output_df.columns.isin(self.expected_cols)).sum() == 0
