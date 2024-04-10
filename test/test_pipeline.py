import logging
import os
import shutil
import sys
import unittest

import pandas as pd

from night_horizons import pipeline
from night_horizons.utils import StdoutLogger, deep_merge

# Configure logging
LOGGER = logging.getLogger(__name__)


class TestStage(unittest.TestCase):

    def setUp(self):

        # Start saving the log
        sys.stdout = StdoutLogger(LOGGER, sys.stdout)

        self.output_dir = './test/test_data/temp'
        self.default_local_options = {
            'io_manager': {
                'output_dir': self.output_dir,
            },
        }

        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Check for other output dirs as well
        for i in range(10):
            other_output_dir = f'{self.output_dir}_v{i:03d}'
            if os.path.isdir(other_output_dir):
                shutil.rmtree(other_output_dir)

    def tearDown(self):

        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Check for other output dirs as well
        for i in range(10):
            other_output_dir = f'{self.output_dir}_v{i:03d}'
            if os.path.isdir(other_output_dir):
                shutil.rmtree(other_output_dir)

    def create_stage(self, config_path, local_options={}):
        '''Wrapper for create_stage so we can direct the output to a
        temporary directory.
        '''

        local_options = {**self.default_local_options, **local_options}

        stage = pipeline.create_stage(
            config_path,
            local_options=local_options
        )

        return stage

    def check_output(self, stage, skip_keys=[]):

        io_manager = stage.container.get_service('io_manager')

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


class TestMetadataProcessor(TestStage):

    def test_metadata_processor(self):

        metadata_processor: pipeline.MetadataProcessor = \
            self.create_stage('./configs/metadata.yaml')
        X_out = metadata_processor.run()

        # Check we found some images
        io_manager = metadata_processor.container.get_service('io_manager')
        assert len(io_manager.input_filepaths['images']) > 0

        # Test for existence
        self.check_output(metadata_processor)

        # Check also that the dataframe is not empty
        df = pd.read_csv(io_manager.output_filepaths['metadata'])
        self.assertFalse(df.empty)

        # Check that the config was saved and can be used to create a duplicate
        stage = self.create_stage(io_manager.output_filepaths['used_config'])
        X_out2 = stage.run()
        assert X_out.equals(X_out2)

        os.makedirs('./configs/generated_templates', exist_ok=True)
        shutil.copy(
            io_manager.output_filepaths['used_config'],
            './configs/generated_templates/metadata.yaml'
        )


class TestMosaicMaker(TestStage):

    def test_mosaicmaker(self):

        mosaicmaker: pipeline.MosaicMaker = \
            self.create_stage('./configs/mosaic.yaml')
        X_out = mosaicmaker.run()

        # Check basic structure of X_out
        io_manager = mosaicmaker.container.get_service('io_manager')
        self.assertEqual(
            len(X_out),
            len(io_manager.input_filepaths['referenced_images'])
        )

        # Check rest of output
        self.check_output(mosaicmaker)

        # Check that the config was saved and can be used to create a duplicate
        mosaicmaker2: pipeline.MosaicMaker = \
            self.create_stage(io_manager.output_filepaths['used_config'])
        X_out2 = mosaicmaker2.run()
        assert X_out.equals(X_out2)

        # Copy the config as a template
        os.makedirs('./configs/generated_templates', exist_ok=True)
        shutil.copy(
            io_manager.output_filepaths['used_config'],
            './configs/generated_templates/mosaic.yaml'
        )


class TestSequentialMosaicMaker(TestStage):

    def test_sequential_mosaickmaker(self):

        local_options = {
            'pipeline': {
                'score_output': True,
            },
            'io_manager': {
                'input_description': {
                    # Overwrite config default, which is nadir only
                    'images': {
                        'directory': 'images/220513-FH135',
                    },
                    # Using a test dir is turned off by default
                    'test_referenced_images': {
                        'directory': 'test_referenced_images/220513-FH135',
                    },
                },
                'output_dir': self.output_dir,
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
        local_options['io_manager_train'] = local_options['io_manager']

        mosaicmaker: pipeline.SequentialMosaicMaker = self.create_stage(
            './configs/sequential-mosaic.yaml',
            local_options
        )
        y_pred = mosaicmaker.run()

        # Check the number of successes
        # Only 2 -> the test image and also a copy of it we put in the raw dir
        assert (y_pred['return_code'] == 'success').sum() == 2

        # Test for existence
        self.check_output(mosaicmaker)

        # Check basic structure of X_out
        io_manager = mosaicmaker.container.get_service('io_manager')
        n_raw = len(io_manager.input_filepaths['images'])
        self.assertEqual(
            len(y_pred),
            n_raw + len(io_manager.input_filepaths['test_referenced_images'])
        )

        # Check the score
        avg_score = y_pred.loc[
            y_pred['return_code'] == 'success',
            'score'
        ].mean()
        assert avg_score < 500.

        # Copy the config as a template
        os.makedirs('./configs/generated_templates', exist_ok=True)
        shutil.copy(
            io_manager.output_filepaths['used_config'],
            './configs/generated_templates/sequential-mosaic.yaml'
        )
