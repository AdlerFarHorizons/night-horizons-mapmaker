from abc import ABC, abstractmethod
import os
import shutil
from typing import Tuple

import argparse
import cv2
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
import sys
from pyproj import CRS

from night_horizons.transformers import filters, order, preprocessors, raster

from night_horizons.container import DIContainer
from night_horizons.io_manager import (
    IOManager, MosaicIOManager, TrainMosaicIOManager
)
from night_horizons.image_processing import (
    mosaicking, operators, processors, registration, scorers
)
from night_horizons.utils import ReferencedRawSplitter


class Mapmaker:
    def __init__(
        self,
        container: DIContainer,
        score_output: bool = False,
        verbose: bool = True
    ):

        self.container = container
        self.score_output = score_output
        self.verbose = verbose

        self.register_fundamental_services()

        self.register_default_services()

    def register_default_services(self):
        pass

    def register_fundamental_services(self):
        '''Services that are used almost-ubiquitously by others.

        By passing singleton=True we ensure that once an instance is made
        we use it throughout, without needing to explicitly pass it

        Parameters
        ----------
        Returns
        -------
        '''

        self.container.register_service(
            'io_manager',
            MosaicIOManager,
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

    # TODO: Delete
    # def register_default_io(self):

    #     dataio_services = self.container.register_dataio_services()
    #
    #     def register_io_manager(*args, **kwargs):
    #         data_ios = {
    #             name: self.container.get_service(name)
    #             for name in dataio_services
    #         }
    #         return IOManager(data_ios=data_ios, *args, **kwargs)
    #     self.container.register_service('io_manager', register_io_manager)


class MosaicMaker(Mapmaker):

    def run(self):

        if self.verbose:
            print('Starting mosaic creation.')

        # Get the filepaths
        io_manager = self.container.get_service('io_manager')
        referenced_fps = io_manager.input_filepaths['referenced_images']

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
        preprocessor = self.container.get_service('preprocessor')
        X = preprocessor.fit_transform(referenced_fps)

        # Mosaicking
        if self.verbose:
            print('Making mosaic...')
        mosaicker = self.container.get_service('mosaicker')
        X_out = mosaicker.fit_transform(X)

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved at {io_manager.output_filepaths["mosaic"]}'
            )

        return X_out, io_manager

    def register_default_services(self):

        self.register_default_preprocessors()
        self.register_default_processors()

    def register_default_preprocessors(self):

        # Preprocessor to get metadata
        self.container.register_service(
            'metadata_preprocessor',
            lambda *args, **kwargs: preprocessors.NITELitePreprocessor(
                io_manager=self.container.get_service('io_manager'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
        )

        # Preprocessor to get geotiff metadata (which includes georeferencing)
        self.container.register_service(
            'geotiff_preprocessor',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
        )

        # Preprocessor to order images
        self.container.register_service(
            'quality_order',
            lambda quality_col='imuGyroMag', *args, **kwargs:
                order.OrderTransformer(
                    order_columns=quality_col,
                    *args, **kwargs
                )
        )

        # Put it all together
        def make_preprocessor_pipeline(
            steps: list[str] = [
                'geotiff_preprocessor',
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
            'preprocessor',
            make_preprocessor_pipeline,
        )

        # These are the steps for making a finalized, polished referenced image
        hq_steps = [
            'metadata_preprocessor',
            'geotiff_preprocesor',
            'quality_order',
        ]
        self.container.register_service(
            'hq_preprocessor',
            lambda steps=hq_steps, *args, **kwargs: make_preprocessor_pipeline(
                steps=steps,
                *args, **kwargs
            )
        )

    def register_default_processors(self):

        # Standard image operator for mosaickers is just a blender
        self.container.register_service(
            'image_operator',
            operators.ImageBlender,
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
            )
        )

        # This is the operator for scoring images
        self.container.register_service(
            'image_scorer',
            scorers.SimilarityScoreOperator,
        )

        # This is the corresponding processor for scoring images
        self.container.register_service(
            'scorer',
            lambda *args, **kwargs: scorers.DatasetScorer(
                io_manager=self.container.get_service('io_manager'),
                image_operator=self.container.get_service('image_scorer'),
                *args, **kwargs
            )
        )

        # Finally, the mosaicker itself, which is a batch processor
        self.container.register_service(
            'mosaicker',
            lambda *args, **kwargs: mosaicking.Mosaicker(
                io_manager=self.container.get_service('io_manager'),
                processor=self.container.get_service('processor'),
                scorer=self.container.get_service('scorer'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
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
            print('    Preparing unreferenced images...')
        preprocessor = self.container.get_service('preprocessor')
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

        return y_pred, io_manager

    def register_default_services(self):

        self.register_validation_services()

        self.register_default_preprocessors()

        self.register_default_train_services()

        self.register_default_processors()

    def register_validation_services(self):

        # For splitting the data
        self.container.register_service(
            'data_splitter',
            lambda *args, **kwargs: ReferencedRawSplitter(
                io_manager=self.container.get_service('io_manager'),
                random_state=self.container.get_service('random_state'),
                *args, **kwargs
            )
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
            )
        )

    def register_default_preprocessors(self):

        # Preprocessor to get metadata
        self.container.register_service(
            'metadata_preprocessor',
            lambda *args, **kwargs: preprocessors.NITELitePreprocessor(
                io_manager=self.container.get_service('io_manager'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
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
            )
        )

        # Preprocessor to get geotiff metadata (which includes georeferencing)
        self.container.register_service(
            'geotiff_preprocessor',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
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
                'metadata_preprocessor',
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
            'preprocessor',
            make_preprocessor_pipeline,
        )

    def register_default_train_services(self):

        # Preprocessor for the referenced mosaic is just a GeoTIFFPreprocessor
        self.container.register_service(
            'preprocessor_train',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
        )

        # The io manager for the training mosaic uses the same parameters
        # as the main io manager, but with some minor modifications
        self.container.register_service(
            'io_manager_train',
            TrainMosaicIOManager,
            singleton=True,
            args_key='io_manager',
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
            )
        )

        # The actual training mosaicker
        self.container.register_service(
            'mosaicker_train',
            lambda *args, **kwargs: mosaicking.Mosaicker(
                io_manager=self.container.get_service('io_manager_train'),
                processor=self.container.get_service('processor_train'),
                crs=self.container.get_service('crs'),
                *args, **kwargs
            )
        )

    def register_default_processors(self):
        '''
        TODO: This could be cleaned up more, at the cost of flexibility.
        '''

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
            )
        )

        # The processor for the sequential mosaicker
        self.container.register_service(
            'processor',
            lambda *args, **kwargs: processors.DatasetRegistrar(
                io_manager=self.container.get_service('io_manager'),
                image_operator=self.container.get_service('image_operator'),
                *args, **kwargs
            )
        )

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
            )
        )


def create_mapmaker(config_filepath, local_options={}):

    container = DIContainer(
        config_filepath=config_filepath,
        local_options=local_options,
    )

    def mapmaker_constructor(
        container,
        map_type: str = 'base',
        *args, **kwargs
    ):
        if map_type == 'base':
            return Mapmaker(container, *args, **kwargs)
        elif map_type == 'mosaic':
            return MosaicMaker(container, *args, **kwargs)
        elif map_type == 'sequential':
            return SequentialMosaicMaker(container, *args, **kwargs)
        else:
            raise ValueError(f'Unknown mapmaker type: {map_type}')

    container.register_service(
        'mapmaker',
        mapmaker_constructor
    )

    return container.get_service('mapmaker', container=container)


if __name__ == "__main__":

    # Set up the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_filepath',
        type=str,
        help='Location of config file.',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create the mapmaker
    mapmaker = create_mapmaker(args.config_filepath)

    # Execute
    mapmaker.run()
