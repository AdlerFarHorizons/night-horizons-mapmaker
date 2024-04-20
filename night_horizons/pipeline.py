from abc import ABC, abstractmethod
import os
import shutil
from typing import Tuple

import argparse
import cv2
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
import sys
import pandas as pd
from pyproj import CRS

from night_horizons.transformers import filters, order, preprocessors, raster

from night_horizons.container import DIContainer
from night_horizons.io_manager import (
    IOManager, MosaicIOManager, SequentialMosaicIOManager, TrainMosaicIOManager
)
from night_horizons.image_processing import (
    mosaicking, operators, processors, registration, scorers
)
from night_horizons.io_manager import ReferencedRawSplitter


class Stage(ABC):
    '''Class responsible for running one part, or stage, of the pipeline.
    This class
    a) prepares the ingredients for the pipeline (e.g.
        self.register_default_services)
    b) takes the ingredients and uses them (e.g. self.run)
    '''

    def __init__(
        self,
        container: DIContainer,
        stage: str = 'base',
        score_output: bool = False,
        verbose: bool = True
    ):
        '''
        Parameters
        ----------
            container :
                Dependency injection container, i.e. a class responsible for
                creating the objects used in the pipeline.
            score_output :
                If True, score data if scoring is implemented
            verbose :
                If True, print extra details in the output.
        '''

        self.container = container
        # TODO: Delete
        self.stage = stage
        self.score_output = score_output
        self.verbose = verbose

        self.register_fundamental_services()

        self.register_default_services()

    def register_fundamental_services(self):
        '''Prepare classes that are used ubiquitously.

        We sometimes set singleton=True. This ensures that once an instance
        is made we use it throughout, without needing to explicitly specify.
        '''

        self.container.register_service(
            'io_manager',
            IOManager,
            singleton=True,
        )
        self.container.register_service(
            'crs',
            CRS,
            singleton=True,
        )
        self.container.register_service(
            'random_state',
            check_random_state,
            singleton=True,
        )

        # Processors for different formats of metadata
        self.container.register_service(
            'metadata_processor',
            constructor=None
        )
        self.container.register_service(
            'fh135_metadata_processor',
            lambda *args, **kwargs: preprocessors.NITELitePreprocessor135(
                io_manager=self.container.get_service('io_manager'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=preprocessors.NITELitePreprocessor135,
        )
        self.container.register_service(
            'fh145_metadata_processor',
            lambda *args, **kwargs: preprocessors.NITELitePreprocessor145(
                io_manager=self.container.get_service('io_manager'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=preprocessors.NITELitePreprocessor145,
        )

    def register_default_services(self):
        '''Main function for registering services
        (preparing classes for construction)
        '''

    def validate(self):
        '''Check that crucial and fundamental services work,
        and that we have found some input filepaths.
        '''

        print('Validating IO setup...')
        io_manager = self.container.get_service('io_manager')

        print('Counting input filepaths...')
        input_fp_count = {
            key: len(val) for key, val
            in io_manager.input_filepaths.items()
        }
        total_fp_count = 0
        for key, count in input_fp_count.items():
            print(f'    {key}: {count} filepaths')
            total_fp_count += count
        print(f'    ------------\n    Total: {total_fp_count} filepaths')
        if total_fp_count == 0:
            print('WARNING: No input filepaths found.')

    def run(self):
        '''Execute the pipeline stage.
        '''


class MetadataProcessor(Stage):
    '''Class for processing metadata (IMU log, GPS log, image log).
    '''

    def run(self):
        '''Process the metadata.
        '''

        if self.verbose:
            print('Starting metadata processing.')

        # Get the filepaths
        if self.verbose:
            print('    Setting up filetree...')
        io_manager: IOManager = self.container.get_service('io_manager')
        image_fps = io_manager.input_filepaths['images']

        # Save config
        if 'used_config' in io_manager.output_filepaths:
            self.container.save_config(
                io_manager.output_filepaths['used_config'])

        # Run the processing
        if self.verbose:
            print('    Running processing...')
        metadata_processor = self.container.get_service(
            'metadata_processor')
        metadata: pd.DataFrame = metadata_processor.fit_transform(image_fps)

        # Save the output
        if self.verbose:
            print('    Saving output...')
        output_fp = io_manager.output_filepaths['metadata']
        os.makedirs(
            os.path.abspath(os.path.dirname(output_fp)),
            exist_ok=True
        )
        metadata.to_csv(output_fp)

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved at {io_manager.output_filepaths["metadata"]}'
            )

        return metadata


class MosaicMaker(Stage):
    '''Create a mosaic from referenced images.
    '''

    def run(self):
        '''Execute the mosaic creation.
        '''

        if self.verbose:
            print('Starting mosaic creation.')

        # Get the filepaths
        io_manager = self.container.get_service('io_manager')
        referenced_fps = io_manager.input_filepaths['referenced_images']

        # Save config
        if 'used_config' in io_manager.output_filepaths:
            self.container.save_config(
                io_manager.output_filepaths['used_config'])

        if self.verbose:
            print(f'Retrieving input from {io_manager.input_dir}')
            print(f'Saving output in {io_manager.output_dir}')
            print(f'Using {len(referenced_fps)} referenced images.')

        assert len(referenced_fps) > 0, (
            'No referenced images found. Search parameters:\n'
            f'{io_manager.input_description["referenced_images"]}'
        )

        # Preprocessing
        if self.verbose:
            print('Preprocessing...')
        preprocessor = self.container.get_service('preprocessor_pipeline')
        x = preprocessor.fit_transform(referenced_fps)

        # Mosaicking
        if self.verbose:
            print('Making mosaic...')
        mosaicker = self.container.get_service('mosaicker')
        x_out = mosaicker.fit_transform(x)

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved at {io_manager.output_filepaths["mosaic"]}'
            )

        return x_out

    def register_default_services(self):
        '''Prepare the services needed for creating a mosaic.
        This includes a modified IO manager, preprocessors, and processors.
        '''

        # TODO: Delete
        # super().register_default_services()

        # Overwrite the io manager
        self.container.register_service(
            'io_manager',
            MosaicIOManager,
            singleton=True,
        )

        self.register_default_preprocessors()
        self.register_default_processors()

    def register_default_preprocessors(self):

        # Preprocessor to get geotiff metadata (which includes georeferencing)
        self.container.register_service(
            'geotiff_preprocessor',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=preprocessors.GeoTIFFPreprocessor,
        )

        # Preprocessor to order images
        self.container.register_service(
            'quality_order',
            lambda quality_col='imuGyroMag', *args, **kwargs:
                order.OrderTransformer(
                    order_columns=quality_col,
                    *args, **kwargs
                ),
            wrapped_constructor=order.OrderTransformer,
        )

        # Put it all together
        def make_preprocessor_pipeline(
            *args,
            steps: list[str] = None,
            **kwargs
        ):
            if steps is None:
                steps = ['geotiff_preprocessor']
            return Pipeline(
                [
                    (step, self.container.get_service(step))
                    for step in steps
                ],
                *args, **kwargs
            )
        self.container.register_service(
            'preprocessor_pipeline',
            make_preprocessor_pipeline,
            wrapped_constructor=Pipeline,
        )

    def register_default_processors(self):

        # Finally, the mosaicker itself, which is a batch processor
        self.container.register_service(
            'mosaicker',
            lambda *args, **kwargs: mosaicking.Mosaicker(
                io_manager=self.container.get_service('io_manager'),
                processor=self.container.get_service('processor'),
                scorer=self.container.get_service('scorer'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=mosaicking.Mosaicker,
        )

        # The processor is deals with saving and loading, in addition to
        # calling the image_operator.
        # By including io_manager as an an argument, we can promote using
        # the same io_manager throughout
        self.container.register_service(
            'processor',
            lambda *args, **kwargs: processors.DatasetUpdater(
                io_manager=self.container.get_service('io_manager'),
                image_operator=self.container.get_service('image_operator'),
                *args, **kwargs
            ),
            wrapped_constructor=processors.DatasetUpdater,
        )

        # This is the corresponding processor for scoring images
        self.container.register_service(
            'scorer',
            lambda *args, **kwargs: scorers.DatasetScorer(
                io_manager=self.container.get_service('io_manager'),
                image_operator=self.container.get_service('image_scorer'),
                *args, **kwargs
            ),
            wrapped_constructor=scorers.DatasetScorer,
        )

        # Standard image operator for mosaickers is just a blender
        self.container.register_service(
            'image_operator',
            operators.ImageBlender,
        )

        # This is the operator for scoring images
        self.container.register_service(
            'image_scorer',
            scorers.SimilarityScoreOperator,
        )


class SequentialMosaicMaker(MosaicMaker):

    def run(self):

        if self.verbose:
            print('Starting sequential mosaic creation.')

        # Get the file management
        io_manager: IOManager = self.container.get_service('io_manager')
        if self.verbose:
            print(f'Retrieving input from {io_manager.input_dir}')
            print(f'Saving output in {io_manager.output_dir}')

        # Save config
        if 'used_config' in io_manager.output_filepaths:
            self.container.save_config(
                io_manager.output_filepaths['used_config'])

        # Split up the data
        splitter: ReferencedRawSplitter = self.container.get_service(
            'data_splitter')
        fps_train, fps_test, fps = splitter.train_test_production_split()
        if self.verbose:
            print(
                f'    Found {len(fps_train) + len(fps_test)} '
                'georeferenced images.'
            )
            print(f'    Found {len(fps)} images to georeference.')

        # Preprocessing for referenced images
        if self.verbose:
            print('Preprocessing...')
            print('    Preparing referenced images...')
        preprocessor_train = self.container.get_service('preprocessor_train')
        X_train = preprocessor_train.fit_transform(fps_train)
        # For the referenced images there is no difference in the dataframe
        # before and after mosaicking, so y_train = X_train
        y_train = X_train

        # Preprocessing for raw images
        if self.verbose:
            print('    Preparing raw images...')
        preprocessor = self.container.get_service('preprocessor_pipeline')
        # The preprocessor is fit to the training sample
        preprocessor = preprocessor.fit(X=fps_train, y=y_train)
        X = preprocessor.transform(fps)

        # Report on preprocessing
        if self.verbose:
            print(
                'Preprocessing determined what images to actually use.\n'
                f'    Using {len(X_train)} georeferenced images.\n'
                f'    Georeferencing {len(X)} images, '
                f'of which {len(fps_test)} have known references '
                'and can be tested.'
            )

        # Mosaicking
        mosaicker: mosaicking.SequentialMosaicker = \
            self.container.get_service('mosaicker')
        if self.verbose:
            print('Creating starting mosaic...')
        mosaicker = mosaicker.fit(
            X=X,
            X_train=X_train,
        )
        if self.verbose:
            print('Mosaicking unreferenced images...')
        y_pred = mosaicker.predict(X)

        # Score the mosaicked images
        if len(fps_test) > 0 and self.score_output:
            if self.verbose:
                print('Scoring the mosaicked test images...')
            y_test = y_pred.loc[fps_test.index]
            y_test = mosaicker.score(y_test)
            y_pred = y_pred.combine_first(y_test)

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved at {io_manager.output_filepaths["mosaic"]}'
            )

        return y_pred

    def register_default_services(self):

        super().register_default_services()

        # Overwrite the io manager
        self.container.register_service(
            'io_manager',
            SequentialMosaicIOManager,
            singleton=True,
        )

        self.register_default_preprocessors()

        self.register_default_processors()

        self.register_default_train_services()

    def register_default_preprocessors(self):

        # For splitting the data
        self.container.register_service(
            'data_splitter',
            lambda *args, **kwargs: ReferencedRawSplitter(
                io_manager=self.container.get_service('io_manager'),
                random_state=self.container.get_service('random_state'),
                *args, **kwargs
            ),
            wrapped_constructor=ReferencedRawSplitter,
        )

        # Preprocessor to use metadata to georeference
        self.container.register_service(
            'metadata_image_registrar',
            lambda passthrough=['filepath', 'camera_num'], *args, **kwargs: (
                registration.MetadataImageRegistrar(
                    crs=self.container.get_service('crs'),
                    passthrough=passthrough,
                    *args, **kwargs
                )
            ),
            wrapped_constructor=registration.MetadataImageRegistrar,
        )

        # Preprocessor to get geotiff metadata (which includes georeferencing)
        self.container.register_service(
            'geotiff_preprocessor',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=preprocessors.GeoTIFFPreprocessor,
        )

        # Preprocessor to filter on altitude
        self.container.register_service(
            'altitude_filter',
            filters.AltitudeFilter,
        )

        # Preprocessor to filter on steadiness
        self.container.register_service(
            'steady_filter',
            filters.SteadyFilter,
        )

        # Preprocessor to order images
        self.container.register_service(
            'order',
            order.SensorAndDistanceOrder,
        )

        # Put it all together
        # One of the more common changes is to replace metadata_image_registrar
        # with geotiff_preprocessor, for testing.
        def make_preprocessor_pipeline(
            steps: list[str] = [
                'metadata_processor',
                'altitude_filter',
                'steady_filter',
                'metadata_image_registrar',
                'order',
            ],
            *args, **kwargs
        ):
            return Pipeline(
                [
                    (step, self.container.get_service(step))
                    for step in steps
                ],
                *args, **kwargs
            )
        self.container.register_service(
            'preprocessor_pipeline',
            make_preprocessor_pipeline,
            wrapped_constructor=Pipeline
        )

    def register_default_processors(self):
        '''
        TODO: This could be cleaned up more, at the cost of flexibility.
        '''

        # Finally, the mosaicker itself
        self.container.register_service(
            'mosaicker',
            lambda *args, **kwargs: mosaicking.SequentialMosaicker(
                io_manager=self.container.get_service('io_manager'),
                processor=self.container.get_service('processor'),
                mosaicker_train=self.container.get_service('mosaicker_train'),
                scorer=self.container.get_service('scorer'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=mosaicking.SequentialMosaicker,
    )

        # The processor for the sequential mosaicker
        self.container.register_service(
            'processor',
            lambda *args, **kwargs: processors.DatasetRegistrar(
                io_manager=self.container.get_service('io_manager'),
                image_operator=self.container.get_service('image_operator'),
                *args, **kwargs
            ),
            wrapped_constructor=processors.DatasetRegistrar,
        )

        # Our scorer.
        # We default to not using an image operator because that's expensive
        self.container.register_service(
            'scorer',
            lambda *args, **kwargs: scorers.ReferencedImageScorer(
                crs=self.container.get_service('crs'),
                io_manager=self.container.get_service('io_manager'),
                image_operator=None,
                *args, **kwargs
            ),
            wrapped_constructor=scorers.ReferencedImageScorer,
        )

        # Operator for the sequential mosaicker--align and blend
        self.container.register_service(
            'image_operator',
            lambda *args, **kwargs: operators.ImageAlignerBlender(
                image_transformer=self.container.get_service(
                    'image_transformer'),
                feature_detector=self.container.get_service(
                    'feature_detector'),
                feature_matcher=self.container.get_service(
                    'feature_matcher'),
                *args, **kwargs
            ),
            wrapped_constructor=operators.ImageAlignerBlender,
        )

        # Feature detection and matching
        self.container.register_service(
            'image_transformer',
            raster.PassImageTransformer,
        )
        self.container.register_service(
            'feature_detector',
            cv2.AKAZE.create,
        )
        self.container.register_service(
            'feature_matcher',
            cv2.BFMatcher.create,
        )

    def register_default_train_services(self):

        # Preprocessor for the referenced mosaic is just a GeoTIFFPreprocessor
        self.container.register_service(
            'preprocessor_train',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=preprocessors.GeoTIFFPreprocessor,
        )

        # The io manager for the training mosaic uses the same parameters
        # as the main io manager, but with some minor modifications
        self.container.register_service(
            'io_manager_train',
            TrainMosaicIOManager,
            singleton=True,
        )
        # TODO: List all the design patterns I used. Singleton, DI, etc.

        # For the training mosaic
        self.container.register_service(
            'image_operator_train',
            operators.ImageBlender,
        )

        # The processor for the training mosaic
        self.container.register_service(
            'processor_train',
            lambda *args, **kwargs: processors.DatasetUpdater(
                io_manager=self.container.get_service('io_manager_train'),
                image_operator=self.container.get_service(
                    'image_operator_train'),
                *args, **kwargs
            ),
            wrapped_constructor=processors.DatasetUpdater,
        )

        # The actual training mosaicker
        self.container.register_service(
            'mosaicker_train',
            lambda *args, **kwargs: mosaicking.Mosaicker(
                io_manager=self.container.get_service('io_manager_train'),
                processor=self.container.get_service('processor_train'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            ),
            wrapped_constructor=mosaicking.Mosaicker,
        )


class QueryProcessor(Stage):
    '''Class for querying pipeline output (primarily referenced images)
    '''

    def run(self):
        '''Query the metadata, copy the files.
        '''

        if self.verbose:
            print('Starting querying.')

        # Get the filepaths
        if self.verbose:
            print('    Setting up filetree...')
        io_manager: IOManager = self.container.get_service('io_manager')
        image_fps = io_manager.input_filepaths['images']

        # Save config
        if 'used_config' in io_manager.output_filepaths:
            self.container.save_config(
                io_manager.output_filepaths['used_config'])

        # Get the metadata
        if self.verbose:
            print('    Accessing metadata...')
        metadata_processor = self.container.get_service(
            'metadata_processor')
        x: pd.DataFrame = metadata_processor.fit_transform(image_fps)

        # Query
        if self.verbose:
            print('    Querying...')
        query_processor = self.container.get_service('query_processor')
        x_out: pd.DataFrame = query_processor.fit_transform(x)

        # Save the output
        if self.verbose:
            print('    Saving...')
        query_results_dir = os.path.join(
            io_manager.output_dir, 'query_results')
        os.makedirs(
            exist_ok=True
        )
        # Copy selected images
        for ind in x_out.index:
            row = x_out.loc[ind]
            output_fp = os.path.join(
                query_results_dir, os.path.basename(row['filepath']))
            shutil.copy(row['filepath'], output_fp)
        # Save the metadata
        x.to_csv(io_manager.output_filepaths['metadata_for_query'])

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved in {query_results_dir}'
            )

        return x

    def register_default_services(self):
        '''Prepare the services needed for creating a mosaic.
        This includes a modified IO manager, preprocessors, and processors.
        '''

        # Register the query filter
        self.container.register_service(
            'query_processor',
            filters.QueryFilter,
        )


def create_stage(config_filepath, local_options={}):

    container = DIContainer(
        config_filepath=config_filepath,
        local_options=local_options,
    )

    stage = container.config['pipeline']['stage']
    if stage == 'base':
        container.register_service(
            'pipeline',
            Stage
        )
    elif stage == 'metadata_processor':
        container.register_service(
            'pipeline',
            MetadataProcessor
        )
    elif stage == 'mosaicker':
        container.register_service(
            'pipeline',
            MosaicMaker
        )
    elif stage == 'sequential_mosaicker':
        container.register_service(
            'pipeline',
            SequentialMosaicMaker
        )
    elif stage == 'query_processor':
        container.register_service(
            'pipeline',
            QueryProcessor
        )
    else:
        raise ValueError(f'Unknown stage: {stage}')

    return container.get_service('pipeline', container=container)


if __name__ == "__main__":

    # Set up the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_filepath',
        type=str,
        help='Location of config file.',
    )
    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='If True, only validate the config file.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create the mapmaker
    mapmaker = create_stage(args.config_filepath)

    # Execute
    if not args.validate_only:
        mapmaker.run()
    else:
        mapmaker.validate()
