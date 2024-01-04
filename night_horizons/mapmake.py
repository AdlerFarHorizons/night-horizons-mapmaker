import cv2

from .container import DIContainer
from . import io_manager, pipelines, preprocessors
from .image_processing import (
    mosaicking, operators, processors, registration, scorers
)


class Mapmaker:
    def __init__(self, config_filepath: str, local_options: dict = {}):
        self.container = DIContainer(
            config_filepath=config_filepath,
            local_options=local_options,
        )


class MosaicMaker(Mapmaker):

    def __init__(self, config_filepath: str, local_options: dict = {}):

        super().__init__(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        self.register_default_services()

    def run(self):

        # Get the filepaths
        io_manager = self.container.get_service('io_manager')
        referenced_fps = io_manager.filepaths['referenced_images']

        # Preprocessing
        preprocessor = self.container.get_service('preprocessor')
        X = preprocessor.fit_transform(referenced_fps)

        # Mosaicking
        mosaicker = self.container.get_service('mosaicker')
        X_out = mosaicker.fit_transform(X)

    def register_default_services(self):

        # What we use for figuring out where to save and load data
        self.container.register_service(
            'io_manager',
            io_manager.MosaicIOManager,
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
        def make_processor(
            io_manager: io_manager.IOManager = None,
            image_operator: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if image_operator is None:
                image_operator = self.container.get_service('image_operator')
            return processors.DatasetUpdater(
                io_manager=io_manager,
                image_operator=image_operator,
                *args, **kwargs
            )
        self.container.register_service('processor', make_processor)

        # This is the operator for scoring images
        self.container.register_service(
            'image_scorer',
            scorers.SimilarityScorer,
        )

        # And this is the corresponding processor for scoring images
        def make_scorer(
            io_manager: io_manager.IOManager = None,
            image_operator: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if image_operator is None:
                image_operator = self.container.get_service('image_scorer')
            return processors.DatasetScorer(
                io_manager=io_manager,
                image_operator=image_operator,
                *args, **kwargs
            )
        self.container.register_service('scorer', make_scorer)

        # Finally, the mosaicker itself, which is a batch processor
        def make_mosaicker(
            io_manager: io_manager.IOManager = None,
            processor: processors.Processor = None,
            scorer: processors.Processor = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if processor is None:
                processor = self.container.get_service(
                    'processor',
                    io_manager=io_manager,
                )
            if scorer is None:
                scorer = self.container.get_service(
                    'scorer',
                    io_manager=io_manager,
                )
            return mosaicking.Mosaicker(
                io_manager=io_manager,
                processor=processor,
                scorer=scorer,
                *args, **kwargs
            )
        self.container.register_service('mosaicker', make_mosaicker)


class SequentialMosaicMaker(MosaicMaker):

    def run(self):

        settings = self.container.config

        # Get the filepaths
        io_manager = self.container.get_service('io_manager')
        fps_train, fps_test, fps = io_manager.train_test_production_split(
            train_size=settings['train_size'],
            random_state=settings['random_state'],
            use_raw_images=settings['use_raw_images'],
        )

        # Preprocessing
        preprocessor = self.container.get_service('preprocessor')
        X_train = preprocessor.fit_transform(fps_train)
        X = preprocessor.fit_transform(fps)

        preprocessor_y = self.container.get_service('preprocessor_y')
        y_train = preprocessor_y.fit_transform(fps_train)
        y_test = preprocessor_y.fit_transform(fps_test)

        # Mosaicking
        mosaicker = self.container.get_service('mosaicker')
        # TODO: It's unintuitive that we use X=y_train here, and approx_y=X.
        y_pred = mosaicker.fit(
            X=y_train,
            approx_y=X,
        )

        return y_pred

    def register_default_services(self):

        # Register file manager typical for mosaickers
        self.container.register_service(
            'io_manager',
            io_manager.MosaicIOManager,
        )

        self.register_default_preprocessors()

        self.register_default_train_services()

        self.register_default_batch_processor()

    def register_default_preprocessors(self):

        # Preprocessor to get metadata
        self.container.register_service(
            'metadata_preprocessor',
            preprocessors.NITELitePreprocessor,
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
            lambda altitude_column='mAltitude', *args, **kwargs: (
                preprocessors.AltitudeFilter(
                    column=altitude_column,
                    *args, **kwargs
                )
            )
        )

        # Preprocessor to filter on steadiness
        def make_steady_filter(
            gyro_columns=['imuGyroX', 'imuGyroY', 'imuGyroZ'],
            *args, **kwargs
        ):
            return preprocessors.SteadyFilter(
                gyro_columns=gyro_columns,
                *args, **kwargs
            )
        self.container.register_service('steady_filter', make_steady_filter)

        self.container.register_service(
            'order',
            preprocessors.SensorAndDistanceOrder,
        )

        def make_preprocessor_pipeline(
            steps = [
                'metadata_preprocessor',
                'altitude_filter',
                'steady_filter',
                'metadata_image_registrar',
                'order',
            ],
            *args, **kwargs
        ):
            return Pipeline([
                (step, self.container.get_service(step, *args, **kwargs))
                for step in steps
            ])
        self.container.register_service(
            'preprocessor_pipeline',
            make_preprocessor_pipeline,
        )

    def register_default_train_services(self):

        # The io manager
        def make_io_manager_train(
            output_description: dict = {
                'mosaic': 'mosaic.tiff',
                'settings': 'settings_train.yaml',
                'log': 'log_train.yaml',
                'y_pred': 'y_pred_train.csv',
                'progress_images_dir_train': 'progress_images_train',
            },
            file_exists: str = 'pass',
            *args, **kwargs
        ):
            '''TODO: It is a requirement that the mosaic output descriptions
            match between io_manager and io_manager_train, but that is not
            enforced in the code.
            '''
            return self.container.get_service(
                'io_manager',
                output_description=output_description,
                file_exists=file_exists,
                *args, **kwargs
            )
        self.container.register_service(
            'io_manager_train',
            make_io_manager_train,
        )

        # For the training mosaic
        self.container.register_service(
            'image_operator_train',
            operators.ImageBlender,
        )

        # And the row transformer typical for mosaickers
        def make_processor_train(
            io_manager_train: io_manager.IOManager = None,
            image_operator_train: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if image_operator_train is None:
                image_operator_train = self.container.get_service(
                    'image_operator_train')
            if io_manager_train is None:
                io_manager_train = self.container.get_service(
                    'io_manager_train')
            return processors.DatasetUpdater(
                io_manager=io_manager_train,
                image_operator=image_operator_train,
                *args, **kwargs
            )
        self.container.register_service(
            'processor_train',
            make_processor_train,
        )

        def make_mosaicker_train(
            io_manager_train: io_manager.IOManager = None,
            processor_train: processors.Processor = None,
            *args, **kwargs
        ):
            if io_manager_train is None:
                io_manager_train = self.container.get_service(
                    'io_manager_train')
            if processor_train is None:
                processor_train = self.container.get_service(
                    'processor_train',
                    io_manager_train=io_manager_train,
                )
            return mosaicking.Mosaicker(
                io_manager=io_manager_train,
                processor=processor_train,
                *args, **kwargs
            )
        self.container.register_service(
            'mosaicker_train',
            make_mosaicker_train,
        )

    def register_default_batch_processor(self):

        # Feature detection and matching
        self.container.register_service(
            'image_transformer',
            preprocessors.PassImageTransformer,
        )
        self.container.register_service(
            'feature_detector',
            cv2.AKAZE.create,
        )
        self.container.register_service(
            'feature_matcher',
            cv2.BFMatcher.create,
        )

        # Image processing
        def make_image_aligner_blender(
            image_transformer: preprocessors.PassImageTransformer = None,
            feature_detector: cv2.Feature2D = None,
            feature_matcher: cv2.DescriptorMatcher = None,
            *args, **kwargs
        ):
            if image_transformer is None:
                image_transformer = self.container.get_service(
                    'image_transformer')
            if feature_detector is None:
                feature_detector = self.container.get_service(
                    'feature_detector')
            if feature_matcher is None:
                feature_matcher = self.container.get_service(
                    'feature_matcher')
            return operators.ImageAlignerBlender(
                image_transformer=image_transformer,
                feature_detector=feature_detector,
                feature_matcher=feature_matcher,
                *args, **kwargs
            )
        self.container.register_service(
            'image_operator',
            make_image_aligner_blender,
        )

        # And the row transformer used for the sequential mosaicker
        def make_processor(
            io_manager: io_manager.IOManager = None,
            image_operator: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if image_operator is None:
                image_operator = self.container.get_service('image_operator')
            return processors.DatasetUpdater(
                io_manager=io_manager,
                image_operator=image_operator,
                *args, **kwargs
            )
        self.container.register_service(
            'processor',
            make_processor
        )

        # Finally, the mosaicker itself
        def make_mosaicker(
            io_manager: io_manager.IOManager = None,
            processor: processors.Processor = None,
            mosaicker_train: mosaicking.Mosaicker = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if processor is None:
                processor = self.container.get_service('processor')
            if mosaicker_train is None:
                mosaicker_train = self.container.get_service('mosaicker_train')
            return mosaicking.SequentialMosaicker(
                io_manager=io_manager,
                processor=processor,
                mosaicker_train=mosaicker_train,
                *args, **kwargs
            )
        self.container.register_service('mosaicker', make_mosaicker)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Please provide a config file path.")
        sys.exit(1)

    config_filepath = sys.argv[1]

    mapmaker = MosaicMaker(config_filepath=config_filepath)
    mapmaker.run()
