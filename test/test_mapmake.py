import os
import shutil
import unittest

from night_horizons import mapmake


class TestMapmake(unittest.TestCase):

    def setUp(self):

        self.out_dir = './test/temp'

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

    def tearDown(self):

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_mosaicmaker(self):

        local_options = {
            'io_manager': {
                'output_dir': self.out_dir,
            }
        }

        mapmaker = mapmake.MosaicMaker('./test/config.yml', local_options)
        X_out, io_manager = mapmaker.run()

        # Check basic structure of X_out
        self.assertEqual(
            len(X_out),
            len(io_manager.input_filepaths['referenced_images'])
        )

        # Check files exist
        skip_keys = ['y_pred', 'progress_images_dir', 'referenced_images']
        for key, filepath in io_manager.output_filepaths.items():
            if key in skip_keys:
                continue
            self.assertTrue(
                os.path.isfile(filepath),
                'Missing file: {}'.format(key),
            )
