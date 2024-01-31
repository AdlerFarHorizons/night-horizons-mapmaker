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

        # Check basic structure of X_out
        self.assertEqual(
            len(X_out),
            len(io_manager.input_filepaths['raw_images'])
        )

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
            'io_manager': {
                'output_dir': self.out_dir,
            },
        }

        mosaicmaker = mapmake.MosaicMaker('./test/config.yml', local_options)
        X_out, io_manager = mosaicmaker.run()

        skip_keys = ['y_pred', 'progress_images_dir', 'referenced_images']

        self.check_output(X_out, io_manager, skip_keys)

    def test_sequential_mosaickmaker(self):

        local_options = {
            'io_manager': {
                'output_dir': self.out_dir,
            },
            'altitude_filter': {
                # So we don't filter anything out
                'float_altitude': 100.,
            },
            'processor': {
                'save_return_codes': ['bad_det', 'out_of_bounds'],
            }
        }

        mosaicmaker = mapmake.SequentialMosaicMaker(
            './test/config.yml',
            local_options
        )
        y_pred, io_manager = mosaicmaker.run()

        self.check_output(
            y_pred,
            io_manager,
        )
