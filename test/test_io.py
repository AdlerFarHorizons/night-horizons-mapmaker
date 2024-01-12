import os
import shutil
import unittest

from night_horizons.io_manager import IOManager


class TestInput(unittest.TestCase):

    def setUp(self):

        self.input_dir = './test/test_data'
        self.output_dir = './test/test_data/mosaics/temp'

    def test_find_files(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'raw_images': {'directory': 'images'},
                'test': 'this/dir.txt',
            },
            output_dir=self.output_dir,
            output_description={},
        )

        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = io_manager.input_filepaths['raw_images']
        assert list(fps) == expected_fps

        assert io_manager.input_filepaths['test'] == \
            './test/test_data/this/dir.txt'

    def test_find_files_exts(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'raw_images': {
                    'directory': 'images',
                    'extension': 'raw',
                },
                'referenced_images': {
                    'directory': 'referenced_images',
                    'extension': ['raw', 'tif', 'tiff'],
                },
            },
            output_dir=self.output_dir,
            output_description={},
        )

        expected_fps_a = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]
        actual_fps_a = io_manager.input_filepaths['raw_images']
        assert list(actual_fps_a) == expected_fps_a

        actual_fps_b = io_manager.input_filepaths['referenced_images']
        assert len(actual_fps_b) > 0

    def test_find_files_pattern(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'referenced_images': {
                    'directory': '',
                    'extension': ['tif', 'tiff'],
                    'pattern': r'Geo\s\d+_\d.tif',
                },
            },
            output_dir=self.output_dir,
            output_description={},
        )

        expected_fps = [
            './test/test_data/referenced_images/Geo 843083290_1.tif',
            './test/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = io_manager.input_filepaths['referenced_images']
        assert list(fps) == expected_fps


class TestOutput(unittest.TestCase):

    def setUp(self):

        self.input_dir = './test/test_data'
        self.output_dir = './test/test_data/mosaics/temp'

        # Start with a clean slate
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):

        # Don't leave a trace
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_prepare_filetree(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={
                'mosaic': 'mosaic.tiff',
                'temp_file': {
                    'filename': 'temp_file.csv',
                    'type': 'csv'
                }
            },
            file_exists='error',
        )

        assert os.path.exists(self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir, 'checkpoints'))

    def test_prepare_filetree_error(self):

        filepath = os.path.join(self.output_dir, 'mosaic.tiff')
        os.makedirs(self.output_dir)
        open(filepath, 'w').close()

        with self.assertRaises(FileExistsError):
            io_manager = IOManager(
                input_dir=self.input_dir,
                input_description={},
                output_dir=self.output_dir,
                output_description={
                    'mosaic': 'mosaic.tiff',
                    'temp_file': {
                        'filename': 'temp_file.csv',
                        'type': 'csv'
                    }
                },
                file_exists='error',
            )

    def test_prepare_filetree_overwrite(self):

        filepath = os.path.join(self.output_dir, 'mosaic.tiff')
        os.makedirs(self.output_dir)
        open(filepath, 'w').close()

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={
                'mosaic': 'mosaic.tiff',
                'temp_file': {
                    'filename': 'temp_file.csv',
                    'type': 'csv'
                }
            },
            file_exists='overwrite',
        )

        assert os.path.exists(self.output_dir)
        assert not os.path.exists(filepath)

    def test_prepare_filetree_new(self):

        filepath = os.path.join(self.output_dir, 'mosaic.tiff')
        os.makedirs(self.output_dir)
        open(filepath, 'w').close()
        new_outdir = './test/test_data/mosaics/temp_v000'

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={
                'mosaic': 'mosaic.tiff',
                'temp_file': {
                    'filename': 'temp_file.csv',
                    'type': 'csv'
                }
            },
            file_exists='new',
        )

        assert os.path.exists(self.output_dir)
        assert os.path.exists(new_outdir)

    def test_test_search_for_checkpoint(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={
                'mosaic': 'mosaic.tiff',
                'temp_file': {
                    'filename': 'temp_file.csv',
                    'type': 'csv'
                },
            },
            file_exists='new',
        )

        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        filepath = os.path.join(checkpoint_dir, 'mosaic_i000013.tiff')
        other_filepath = os.path.join(checkpoint_dir, 'mosaic_i000003.tiff')
        open(filepath, 'w').close()
        open(other_filepath, 'w').close()

        i = io_manager.search_for_checkpoint('mosaic')
        assert i == 13 + 1
