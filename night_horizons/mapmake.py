from abc import ABC, abstractmethod
import os
import shutil

import cv2
from osgeo import gdal
from sklearn.pipeline import Pipeline

from night_horizons.transformers import filters, order, preprocessors, raster

from night_horizons.container import DIContainer
from night_horizons.io_manager import (
    IOManager, MosaicIOManager, TrainMosaicIOManager
)
from night_horizons.image_processing import (
    mosaicking, operators, processors, registration, scorers
)


class Mapmaker:
    def __init__(
        self,
        config_filepath: str,
        local_options: dict = {},
        verbose: bool = True
    ):

        gdal.UseExceptions()

        self.verbose = verbose

        self.container = DIContainer(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        self.register_default_services()

    def register_default_services(self):

        # Services for input/output
        self.container.register_service(
            'io_manager',
            IOManager,
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
            print(f'Saving output in {io_manager.output_dir}')

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

        # Services for input/output
        # By passing singleton=True we ensure that once an instance is made
        # we use it throughout, without needing to explicitly pass it
        self.container.register_service(
            'io_manager',
            MosaicIOManager,
            singleton=True,
        )

        # What we use for preprocessing
        self.container.register_service(
            'preprocessor',
            preprocessors.GeoTIFFPreprocessor
        )

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
            scorers.SimilarityScorer,
        )

        # This is the corresponding processor for scoring images
        self.container.register_service(
            'scorer',
            lambda *args, **kwargs: processors.DatasetScorer(
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
                *args, **kwargs
            )
        )


class SequentialMosaicMaker(MosaicMaker):

    def run(self):

        if self.verbose:
            print('Starting mosaic creation.')

        # Get the filepaths
        io_manager: IOManager = self.container.get_service('io_manager')
        fps_train = io_manager.input_filepaths['referenced_images']
        fps = io_manager.input_filepaths['raw_images']

        if self.verbose:
            print(f'Saving output in {io_manager.output_dir}')

        # Preprocessing
        if self.verbose:
            print('Preprocessing...')

        # Y preprocessing
        if self.verbose:
            print('    Preparing referenced images...')
        preprocessor_y = self.container.get_service('preprocessor_y')
        y_train = preprocessor_y.fit_transform(fps_train)

        # X preprocessing
        if self.verbose:
            print('    Preparing unreferenced images...')
        preprocessor = self.container.get_service('preprocessor')
        preprocessor = preprocessor.fit(X=fps_train, y=y_train)
        X = preprocessor.transform(fps)

        # First guess at image registration
        y_pred_estimate = X[['filepath'] + preprocessors.GEOTRANSFORM_COLS]

        # Mosaicking
        mosaicker: mosaicking.SequentialMosaicker = \
            self.container.get_service('mosaicker')
        if self.verbose:
            print('Creating starting mosaic...')
        mosaicker = mosaicker.fit(
            X=None,
            y=y_train,
            y_pred_estimate=y_pred_estimate,
        )
        if self.verbose:
            print('Mosaicking unreferenced images...')
        y_pred = mosaicker.predict(X)

        if self.verbose:
            print(
                'Done!\n'
                f'Output saved at {io_manager.output_filepaths["mosaic"]}'
            )

        return y_pred, io_manager

    def register_default_services(self):

        # Services for input/output
        self.container.register_service(
            'io_manager',
            MosaicIOManager,
            singleton=True,
        )

        self.register_default_preprocessors()

        self.register_default_train_services()

        self.register_default_batch_processor()

    def register_default_preprocessors(self):

        # Preprocessor to get metadata
        self.container.register_service(
            'metadata_preprocessor',
            lambda *args, **kwargs: preprocessors.NITELitePreprocessor(
                io_manager=self.container.get_service('io_manager'),
                *args, **kwargs
            )
        )

        # Preprocessor to use metadata to georeference
        self.container.register_service(
            'metadata_image_registrar',
            lambda passthrough=['filepath', 'camera_num'], *args, **kwargs: (
                registration.MetadataImageRegistrar(
                    passthrough=passthrough,
                    *args, **kwargs
                )
            )
        )

        # Preprocessor to get geotiff metadata (which includes georeferencing)
        self.container.register_service(
            'geotiff_preprocessor',
            preprocessors.GeoTIFFPreprocessor
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

        # Preprocessor for the y values is just a geotiff preprocessor
        self.container.register_service(
            'preprocessor_y',
            lambda *args, **kwargs: preprocessors.GeoTIFFPreprocessor(
                *args, **kwargs
            )
        )

    def register_default_train_services(self):

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
                *args, **kwargs
            )
        )

    def register_default_batch_processor(self):
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

        # We'll include a scorer as well
        self.container.register_service(
            'image_scorer',
            scorers.SimilarityScorer,
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
                *args, **kwargs
            )
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Please provide a config file path.")
        sys.exit(1)

    mosaic_type = sys.argv[1]
    config_filepath = sys.argv[2]

    if mosaic_type == 'simple':
        mapmaker = MosaicMaker(config_filepath=config_filepath)
    elif mosaic_type == 'sequential':
        mapmaker = SequentialMosaicMaker(config_filepath=config_filepath)
    else:
        print("Please provide a valid mosaic type.")
        sys.exit(1)

    # Execute
    mapmaker.run()
