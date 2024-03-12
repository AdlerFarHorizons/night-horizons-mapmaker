import os
import shutil
import unittest

import pandas as pd

from night_horizons import pipeline

###############################################################################
# DEBUG
###############################################################################
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='./test/test_data/test_log.log',
    filemode='w',
    format='%(message)s'
)
logger = logging.getLogger()


class StdoutLogger(object):
    def __init__(self, logger, stdout):
        self.logger = logger
        self.stdout = stdout

    def write(self, message):
        if message.strip() != "":
            self.logger.info(message.strip())
        self.stdout.write(message)

    def flush(self):
        self.stdout.flush()


# Redirect stdout
sys.stdout = StdoutLogger(logger, sys.stdout)
###############################################################################


class TestStage(unittest.TestCase):

    def setUp(self):

        self.output_dir = './test/test_data/temp'
        self.default_local_options = {
            'io_manager': {
                'output_dir': self.output_dir,
            },
        }

        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):

        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

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
            self.create_stage('./configs/metadata.yml')
        X_out = metadata_processor.run()

        # Check we found some images
        io_manager = metadata_processor.container.get_service('io_manager')
        assert len(io_manager.input_filepaths['images']) > 0

        # Test for existence
        self.check_output(metadata_processor)

        # Check also that the dataframe is not empty
        df = pd.read_csv(io_manager.output_filepaths['metadata'])
        self.assertFalse(df.empty)


class TestMosaicMaker(TestStage):

    def test_mosaicmaker(self):

        mosaicmaker: pipeline.MosaicMaker = \
            self.create_stage('./configs/mosaic.yml')
        X_out = mosaicmaker.run()

        # Check basic structure of X_out
        io_manager = mosaicmaker.container.get_service('io_manager')
        self.assertEqual(
            len(X_out),
            len(io_manager.input_filepaths['referenced_images'])
        )

        # Check rest of output
        self.check_output(mosaicmaker)


class TestSequentialMosaicMaker(TestStage):

    def test_sequential_mosaickmaker(self):
        '''TODO: This fails some fraction of the time, with the Python process
        being killed.
        '''

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

        mosaicmaker: pipeline.SequentialMosaicMaker = self.create_stage(
            './configs/sequential-mosaic.yml',
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

# TODO: TestScoreSequentialMosaicker
