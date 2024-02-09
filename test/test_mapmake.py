import os
import shutil
import unittest

from night_horizons import mapmake


class TestMapmake(unittest.TestCase):

    def setUp(self):

        self.out_dir = './test/test_data/temp'

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

    def tearDown(self):

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

    def check_output(self, X_out, io_manager, skip_keys=[]):

        # Check files exist
        for key, filepath in io_manager.output_filepaths.items():
            if key in skip_keys:
                continue
            if not (os.path.isdir(filepath) or os.path.isfile(filepath)):
                if '{' in filepath:
                    listed = os.listdir(os.path.dirname(filepath))
                    if len(listed) == 0:
                        raise AssertionError(
                            f'Missing file(s), {key}: {filepath}'
                        )
                else:
                    raise AssertionError(
                        f'Missing file, {key}: {filepath}'
                    )

    def test_mosaicmaker(self):

        local_options = {
            'mapmaker': {'map_type': 'mosaic'},
            'io_manager': {'output_dir': self.out_dir},
        }

        mosaicmaker = mapmake.create_mapmaker(
            './test/config.yml',
            local_options,
        )
        X_out, io_manager = mosaicmaker.run()

        skip_keys = ['y_pred', 'progress_images_dir', 'referenced_images']

        # Check basic structure of X_out
        self.assertEqual(
            len(X_out),
            len(io_manager.input_filepaths['referenced_images'])
        )

        self.check_output(X_out, io_manager, skip_keys)

    def test_sequential_mosaickmaker(self):
        '''TODO: This fails some fraction of the time, with the Python process
        being killed.
        '''

        local_options = {
            'mapmaker': {
                'map_type': 'sequential',
                'score_output': True,
            },
            'io_manager': {
                'output_dir': self.out_dir,
            },
            'data_splitter': {
                'use_test_dir': True,
            },
            'altitude_filter': {
                # So we don't filter anything out
                'float_altitude': 100.,
            },
            'processor': {
                'save_return_codes': ['bad_det', 'out_of_bounds'],
            },
        }

        mosaicmaker: mapmake.SequentialMosaicMaker = mapmake.create_mapmaker(
            './test/config.yml',
            local_options
        )
        y_pred, io_manager = mosaicmaker.run()

        # Check basic structure of X_out
        n_raw = len(io_manager.input_filepaths['raw_images'])
        self.assertEqual(
            len(y_pred),
            n_raw + len(io_manager.input_filepaths['test_images'])
        )

        # Check the number of successes
        # Only 2 -> the test image and also a copy of it we put in the raw dir
        assert (y_pred['return_code'] == 'success').sum() == 2

        self.check_output(y_pred, io_manager)

        # Check the score
        avg_score = y_pred.loc[
            y_pred['return_code'] == 'success',
            'score'
        ].mean()
        assert avg_score < 500.
