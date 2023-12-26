'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd

import night_horizons.preprocessers as preprocessers
import night_horizons.utils as utils


class TestNITELitePreprocesser(unittest.TestCase):

    def setUp(self):

        # Metadata filetree info
        metadata_dir = './test/test_data/metadata'
        self.img_log_fp = os.path.join(metadata_dir, 'image.log')
        self.imu_log_fp = os.path.join(metadata_dir, 'PresIMULog.csv')
        self.gps_log_fp = os.path.join(metadata_dir, 'GPSLog.csv')

        # Preprocesser construction
        self.expected_cols = ['filepath', 'sensor_x', 'sensor_y']
        self.transformer = preprocessers.NITELitePreprocesser(
            output_columns=self.expected_cols
        )

    def test_output(self):
        '''The output prior to any form of georeferencing.
        '''

        # Image filetree info
        image_dir = './test/test_data/images'
        fps = utils.discover_data(image_dir)
        n_files = len(fps)

        # Add test columns to passthrough
        fps = utils.check_filepaths_input(fps)
        fps['test_column'] = 1
        fps['sensor_x'] = np.nan
        self.transformer.passthrough = ['test_column', 'sensor_x']

        metadata = self.transformer.fit_transform(
            fps,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )
        assert len(metadata) == n_files
        utils.check_columns(
            actual=metadata.columns,
            expected=self.expected_cols,
            passthrough=self.transformer.passthrough,
        )
        assert metadata['sensor_x'].isna().sum() == 0

    def test_output_referenced_files(self):

        # Image filetree info
        image_dir = './test/test_data/referenced_images'
        fps = utils.discover_data(image_dir)
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

    def test_output_no_file_found_but_keep_it_around(self):
        '''The output prior to any form of georeferencing.
        '''

        self.transformer.unhandled_files = 'warn and passthrough'

        # Image filetree info
        image_dir = './test/test_data/referenced_images'
        fps = utils.discover_data(image_dir)
        n_files = len(fps)
        fps = pd.concat([pd.Series(['not_a_file']), fps], ignore_index=True)

        # Scramble the indices, so we know this works regardless of the input
        rng = np.random.default_rng(15213)
        new_index = rng.choice(np.arange(100), size=len(fps), replace=False)
        fps.index = new_index

        metadata = self.transformer.fit_transform(
            fps,
            img_log_fp=self.img_log_fp,
            imu_log_fp=self.imu_log_fp,
            gps_log_fp=self.gps_log_fp,
        )
        assert len(metadata) == n_files + 1
        assert (~metadata.columns.isin(self.expected_cols)).sum() == 0
        assert metadata['sensor_x'].isna().sum() == 1

        # Check that the order is not garbled.
        assert (metadata['filepath'] != fps).sum() == 0
        np.testing.assert_allclose(metadata.index, fps.index)


class TestGeoTIFFPreprocesser(unittest.TestCase):

    def test_output(self):

        # Image filetree info
        image_dir = './test/test_data/images'
        fps = utils.discover_data(image_dir)
        n_files_unreffed = len(fps)
        referenced_image_dir = './test/test_data/referenced_images'
        fps2 = utils.discover_data(
            referenced_image_dir,
            extension=['tif', 'tiff']
        )
        fps = pd.concat([fps, fps2], ignore_index=True)
        n_files = len(fps)

        transformer = preprocessers.GeoTIFFPreprocesser()
        X = transformer.fit_transform(fps)

        expected_cols = ['filepath',] + preprocessers.GEOTRANSFORM_COLS
        assert (~X.columns.isin(expected_cols)).sum() == 0

        assert len(X) == n_files
        assert X['x_min'].isna().sum() == n_files_unreffed

        # Ensure the conversion was done, i.e. there shouldn't really be
        # values to zero, unless our test dataset was within 1000m of (0,0)
        assert np.nanmax(np.abs(X['x_min'])) > 1000
