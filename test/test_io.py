import os
import shutil
import unittest

from night_horizons import io_manager


class TestInputFileManager(unittest.TestCase):

    def setUp(self):

        self.in_dir = './test/test_data'

    def test_find_files(self):

        file_manager_in = io_manager.InputFileManager(
            in_dir=self.in_dir,
            raw_images={'directory': 'images'},
        )

        expected_fps = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]

        fps = file_manager_in.find_files('raw_images')
        assert list(fps) == expected_fps

    def test_find_files_exts(self):

        file_manager_in = io_manager.InputFileManager(
            in_dir=self.in_dir,
            **{
                'raw_images': {
                    'directory': 'images',
                    'extension': 'raw',
                },
                'referenced_images': {
                    'directory': 'referenced_images',
                    'extension': ['raw', 'tif', 'tiff'],
                },
            }
        )

        expected_fps_a = [
            ('./test/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('./test/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
        ]
        actual_fps_a = file_manager_in.find_files('raw_images')
        assert list(actual_fps_a) == expected_fps_a

        actual_fps_b = file_manager_in.find_files('referenced_images')
        assert len(actual_fps_b) > 0

    def test_find_files_pattern(self):

        file_manager_in = io_manager.InputFileManager(
            in_dir=self.in_dir,
            referenced_images={
                'directory': '',
                'extension': ['tif', 'tiff'],
                'pattern': r'Geo\s\d+_\d.tif',
            },
        )

        expected_fps = [
            './test/test_data/referenced_images/Geo 843083290_1.tif',
            './test/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = file_manager_in.find_files('referenced_images')
        assert list(fps) == expected_fps


class TestOutputFileManager(unittest.TestCase):

    def setUp(self):

        self.out_dir = './test/test_data/mosaics/temp'

        # Start with a clean slate
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.file_manager_out = io_manager.OutputFileManager(
            out_dir=self.out_dir,
            filename='mosaic.tiff',
            file_exists='error',
        )

    def tearDown(self):

        # Don't leave a trace
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_prepare_filetree(self):

        self.file_manager_out.prepare_filetree()
        assert os.path.exists(self.out_dir)
        assert os.path.exists(os.path.join(self.out_dir, 'checkpoints'))

    def test_prepare_filetree_error(self):

        filepath = os.path.join(self.out_dir, 'mosaic.tiff')
        os.makedirs(self.out_dir)
        open(filepath, 'w').close()

        with self.assertRaises(FileExistsError):
            self.file_manager_out.prepare_filetree()
