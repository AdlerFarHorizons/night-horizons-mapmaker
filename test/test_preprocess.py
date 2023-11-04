'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np

import night_horizons.preprocess as preprocess


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

        fps = preprocess.discover_data(image_dir, ['raw', 'tif', 'tiff'])
        assert fps == expected_fps


class TestNITELitePreprocesser(unittest.TestCase):

    def setUp(self):

        # Metadata filetree info
        metadata_dir = './test/test_data/metadata'
        self.img_log_fp = os.path.join(metadata_dir, 'image.log')
        self.imu_log_fp = os.path.join(metadata_dir, 'PresIMULog.csv')
        self.gps_log_fp = os.path.join(metadata_dir, 'GPSLog.csv')

        # Preprocesser construction
        self.expected_cols = ['filepath', 'sensor_x', 'sensor_y']
        self.transformer = preprocess.NITELitePreprocesser(
            output_columns=self.expected_cols
        )

    def test_output(self):
        '''The output prior to any form of georeferencing.
        '''

        # Image filetree info
        image_dir = './test/test_data/images'
        fps = preprocess.discover_data(image_dir)
        n_files = len(fps)

        metadata = self.transformer.fit_transform(
            fps,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )
        assert len(metadata) == n_files
        assert (~metadata.columns.isin(self.expected_cols)).sum() == 0
        assert metadata['sensor_x'].isna().sum() == 0

    def test_output_referenced_files(self):

        # Image filetree info
        image_dir = './test/test_data/referenced_images'
        fps = preprocess.discover_data(image_dir)
        n_files = len(fps)

        metadata = self.transformer.fit_transform(
            fps,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )
        assert len(metadata) == n_files
        assert (~metadata.columns.isin(self.expected_cols)).sum() == 0
        assert metadata['sensor_x'].isna().sum() == 0


class TestGeoTIFFPreprocesser(unittest.TestCase):

    def test_output(self):

        # Image filetree info
        image_dir = './test/test_data/images'
        fps = preprocess.discover_data(image_dir)
        n_files_unreffed = len(fps)
        referenced_image_dir = './test/test_data/referenced_images'
        fps.extend(preprocess.discover_data(
            referenced_image_dir,
            extension=['tif', 'tiff']
        ))
        n_files = len(fps)

        transformer = preprocess.GeoTIFFPreprocesser()
        X = transformer.fit_transform(fps)

        expected_cols = ['filepath',] + preprocess.GEOTRANSFORM_COLS
        assert (~X.columns.isin(expected_cols)).sum() == 0

        assert len(X) == n_files
        assert X['x_min'].isna().sum() == n_files_unreffed

        # Ensure the conversion was done, i.e. there shouldn't really be
        # values to zero, unless our test dataset was within 1000m of (0,0)
        assert np.nanmax(np.abs(X['x_min'])) > 1000
