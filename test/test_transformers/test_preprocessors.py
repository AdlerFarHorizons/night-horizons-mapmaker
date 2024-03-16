'''Test file for preprocessing.
'''

import os
import shutil
import unittest

import numpy as np
import pandas as pd

import night_horizons.transformers.preprocessors as preprocessors
import night_horizons.utils as utils
from night_horizons.pipeline import create_stage


class TestNITELitePreprocessor(unittest.TestCase):

    def setUp(self):

        self.mapmaker = create_stage('./test/config.yml')
        self.io_manager = self.mapmaker.container.get_service('io_manager')

        # Preprocessor construction
        self.expected_cols = ['filepath', 'sensor_x', 'sensor_y']
        self.transformer = preprocessors.NITELitePreprocessor(
            io_manager=self.io_manager,
            output_columns=self.expected_cols,
            crs=self.mapmaker.container.get_service('crs'),
        )

        if os.path.isdir(self.io_manager.output_dir):
            shutil.rmtree(self.io_manager.output_dir)

    def tearDown(self):

        if os.path.isdir(self.io_manager.output_dir):
            shutil.rmtree(self.io_manager.output_dir)

    def test_output(self):
        '''The output prior to any form of georeferencing.
        '''

        # Image filetree info
        image_dir = '/data/input/images'
        fps = utils.discover_data(image_dir)
        n_files = len(fps)

        # Add test columns to passthrough
        fps = utils.check_filepaths_input(fps)
        fps['test_column'] = 1
        fps['sensor_x'] = np.nan
        self.transformer.passthrough = ['test_column', 'sensor_x']

        metadata = self.transformer.fit_transform(fps)
        assert len(metadata) == n_files
        utils.check_columns(
            actual=metadata.columns,
            expected=self.expected_cols,
            passthrough=self.transformer.passthrough,
        )
        assert metadata['sensor_x'].isna().sum() == 0

        # Test that we can reload it once saved
        output_fp = os.path.join(self.io_manager.output_dir, 'metadata.csv')
        os.makedirs(self.io_manager.output_dir, exist_ok=True)
        metadata.to_csv(output_fp)
        self.io_manager.input_filepaths['metadata'] = output_fp
        del self.io_manager.input_filepaths['img_log']
        del self.io_manager.input_filepaths['gps_log']
        del self.io_manager.input_filepaths['imu_log']
        metadata2 = self.transformer.fit_transform(fps)

        assert metadata.equals(metadata2)

    def test_output_referenced_files(self):

        # Image filetree info
        image_dir = '/data/input/referenced_images'
        fps = utils.discover_data(image_dir)
        n_files = len(fps)

        metadata = self.transformer.fit_transform(fps)
        assert len(metadata) == n_files - 1
        assert (~metadata.columns.isin(self.expected_cols)).sum() == 0
        assert metadata['sensor_x'].isna().sum() == 0

    def test_output_no_file_found_but_keep_it_around(self):
        '''The output prior to any form of georeferencing.
        '''

        self.transformer.unhandled_files = 'passthrough'

        # Image filetree info
        image_dir = '/data/input/referenced_images'
        fps = utils.discover_data(image_dir)
        n_files = len(fps)
        fps = pd.concat([pd.Series(['not_a_file']), fps], ignore_index=True)

        # Scramble the indices, so we know this works regardless of the input
        rng = np.random.default_rng(15213)
        new_index = rng.choice(np.arange(100), size=len(fps), replace=False)
        fps.index = new_index

        metadata = self.transformer.fit_transform(fps)
        assert len(metadata) == n_files + 1
        assert (~metadata.columns.isin(self.expected_cols)).sum() == 0
        assert metadata['sensor_x'].isna().sum() == 2

        # Check that the order is not garbled.
        assert (metadata['filepath'] != fps).sum() == 0
        np.testing.assert_allclose(metadata.index, fps.index)


class TestGeoTIFFPreprocessor(unittest.TestCase):

    def test_output(self):
        '''TODO: This fails when run alongside the other tests in
        test_preprocessors, but not by itself.
        '''

        # Image filetree info
        image_dir = '/data/input/images'
        raw_fps = utils.discover_data(image_dir, extension=['raw'])
        n_files_unreffed = len(raw_fps)
        referenced_image_dir = \
            '/data/input/referenced_images'
        referenced_fps = utils.discover_data(
            referenced_image_dir,
            extension=['tif', 'tiff']
        )
        fps = pd.concat([raw_fps, referenced_fps], ignore_index=True)
        n_files = len(fps)

        transformer = preprocessors.GeoTIFFPreprocessor()
        X = transformer.fit_transform(fps)

        expected_cols = ['filepath',] + preprocessors.GEOTRANSFORM_COLS
        assert (~X.columns.isin(expected_cols)).sum() == 0

        assert len(X) == n_files
        assert X['x_min'].isna().sum() == n_files_unreffed

        # Ensure the conversion was done, i.e. there shouldn't really be
        # values to zero, unless our test dataset was within 1000m of (0,0)
        assert np.nanmax(np.abs(X['x_min'])) > 1000
