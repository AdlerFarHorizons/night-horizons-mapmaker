import os
import shutil
import unittest
import difflib
import pprint

import numpy as np
import pandas as pd

from night_horizons.io_manager import IOManager, ReferencedRawSplitter


def assert_sorted_lists_equal(list1, list2):
    assert sorted(list1) == sorted(list2), (
        "Lists differ:\n" + '\n'.join(difflib.ndiff(
            pprint.pformat(list1).splitlines(),
            pprint.pformat(list2).splitlines())
        )
    )


class TestInput(unittest.TestCase):

    def setUp(self):

        self.input_dir = '/data/input'
        self.output_dir = '/data/output/mosaics/temp'

        self.expected_fps_raw = [
            ('/data/input/images/220513-FH135/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('/data/input/images/220513-FH135/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
            ('/data/input/images/220513-FH135/23085687/'
             'Geo 836109848_1.tif'),
        ]

    def test_find_files(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'images': {'directory': 'images'},
                'test': 'this/dir.txt',
            },
            output_dir=self.output_dir,
            output_description={},
        )

        fps = io_manager.input_filepaths['images']
        assert_sorted_lists_equal(list(fps), self.expected_fps_raw)

        assert io_manager.input_filepaths['test'] == \
            '/data/input/this/dir.txt'

    def test_find_files_exts(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'images': {
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

        actual_fps_a = io_manager.input_filepaths['images']
        assert_sorted_lists_equal(list(actual_fps_a), self.expected_fps_raw[:-1])

        actual_fps_b = io_manager.input_filepaths['referenced_images']
        assert len(actual_fps_b) > 0

    def test_find_files_pattern(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={
                'referenced_images': {
                    'directory': 'referenced_images',
                    'pattern': r'Geo\s\d+_\d.tif$',
                },
            },
            output_dir=self.output_dir,
            output_description={},
        )

        expected_fps = [
            '/data/input/referenced_images/220513-FH135/Geo 843083290_1.tif',
            '/data/input/referenced_images/220513-FH135/Geo 836109848_1.tif',
        ]

        fps = io_manager.input_filepaths['referenced_images']
        assert_sorted_lists_equal(list(fps), expected_fps)


class TestOutput(unittest.TestCase):

    def setUp(self):

        self.input_dir = '/data/input'
        self.output_dir = '/data/output/mosaics/temp'

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
            output_description={'mosaic': 'mosaic.tiff'},
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
                output_description={'mosaic': 'mosaic.tiff'},
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
            output_description={'mosaic': 'mosaic.tiff'},
            file_exists='overwrite',
        )

        assert os.path.exists(self.output_dir)
        assert not os.path.exists(filepath)

    def test_prepare_filetree_new(self):

        filepath = os.path.join(self.output_dir, 'mosaic.tiff')
        os.makedirs(self.output_dir)
        open(filepath, 'w').close()
        new_outdir = '/data/output/mosaics/temp_v000'

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={'mosaic': 'mosaic.tiff'},
            file_exists='new',
        )

        assert os.path.exists(self.output_dir)
        assert os.path.exists(new_outdir)

    def test_test_search_for_checkpoint(self):

        io_manager = IOManager(
            input_dir=self.input_dir,
            input_description={},
            output_dir=self.output_dir,
            output_description={'mosaic': 'mosaic.tiff'},
            file_exists='new',
        )

        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        filepath = os.path.join(checkpoint_dir, 'mosaic_i000013.tiff')
        other_filepath = os.path.join(checkpoint_dir, 'mosaic_i000003.tiff')
        open(filepath, 'w').close()
        open(other_filepath, 'w').close()

        i = io_manager.search_for_checkpoint('mosaic')
        assert i == 13 + 1


class TestReferencedRawSplitter(unittest.TestCase):

    def setUp(self):

        input_dir = '/data/input'
        output_dir = '/data/output/mosaics/temp'
        self.io_manager = IOManager(
            input_dir=input_dir,
            input_description={
                'referenced_images': {
                    'directory': 'referenced_images',
                    'pattern': r'Geo\s\d+_\d.tif$',
                },
                'images': {
                    'directory': 'images',
                    'extension': 'raw',
                },
            },
            output_dir=output_dir,
            output_description={},
        )

    def test_functional(self):

        # Load the data
        test_size = 1
        data_splitter = ReferencedRawSplitter(
            self.io_manager,
            test_size=test_size,
        )
        fps_train, fps_test, fps = data_splitter.train_test_production_split()
        self.assertEqual(len(fps_test), test_size)
