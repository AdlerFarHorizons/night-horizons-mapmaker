from abc import ABC, abstractmethod
import cv2
from sklearn.pipeline import Pipeline

from .transformers import filters, order, preprocessors, raster

from .container import DIContainer
from . import io_manager
from .image_processing import (
    mosaicking, operators, processors, registration, scorers
)


class Mapmaker(ABC):
    def __init__(self, config_filepath: str, local_options: dict = {}):
        self.container = DIContainer(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        self.register_default_services()

    @abstractmethod
    def register_default_services(self):
        pass

    # TODO: Delete
    # def register_default_io(self):

    #     dataio_services = self.container.register_dataio_services()
    #     
    #     def register_io_manager(*args, **kwargs):
    #         data_ios = {
    #             name: self.container.get_service(name)
    #             for name in dataio_services
    #         }
    #         return io_manager.IOManager(data_ios=data_ios, *args, **kwargs)
    #     self.container.register_service('io_manager', register_io_manager)


class MosaicMaker(Mapmaker):

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

        # Services for input/output
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
        # By including io_manager as an an argument, we can promote using
        # the same io_manager throughout
        self.container.register_service(
            'processor',
            lambda io_manager, *args, **kwargs: processors.DatasetUpdater(
                io_manager=io_manager,
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
            lambda io_manager, *args, **kwargs: processors.DatasetScorer(
                io_manager=io_manager,
                image_operator=self.container.get_service('image_scorer'),
                *args, **kwargs
            )
        )

        # Finally, the mosaicker itself, which is a batch processor
        def make_mosaicker(
            io_manager: io_manager.IOManager = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            processor = self.container.get_service(
                'processor', io_manager=io_manager)
            scorer = self.container.get_service(
                'scorer', io_manager=io_manager)
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

        # Y preprocessing
        preprocessor_y = self.container.get_service('preprocessor_y')
        y_train = preprocessor_y.fit_transform(fps_train)
        y_test = preprocessor_y.fit_transform(fps_test)

        # X preprocessing
        preprocessor = self.container.get_service('preprocessor')
        # Fit the pipeline
        preprocessor = preprocessor.fit(
            X=fps_train,
            y=y_train,
            metadata_preprocessor__img_log_fp=(
                io_manager.input_filepaths['img_log']
            ),
            metadata_preprocessor__imu_log_fp=(
                io_manager.input_filepaths['imu_log']
            ),
            metadata_preprocessor__gps_log_fp=(
                io_manager.input_filepaths['gps_log']
            ),
        )
        X_train = preprocessor.fit_transform(fps_train)
        X = preprocessor.fit_transform(fps)

        # Mosaicking
        mosaicker = self.container.get_service('mosaicker')
        # TODO: It's unintuitive that we use X=y_train here, and approx_y=X.
        mosaicker = mosaicker.fit(
            X=y_train,
            approx_y=X,
        )
        y_pred = mosaicker.fit_transform(X)

        return y_pred

    def register_default_services(self):

        # Services for input/output
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

        # The processor for the training mosaic
        self.container.register_service(
            'processor_train',
            lambda io_manager_train, *args, **kwargs:
                processors.DatasetUpdater(
                    io_manager=io_manager_train,
                    image_operator=self.container.get_service(
                        'image_operator_train'),
                    *args, **kwargs
                )
        )

        # The actual training mosaicker
        def make_mosaicker_train(
            io_manager_train: io_manager.IOManager = None,
            *args, **kwargs
        ):
            if io_manager_train is None:
                io_manager_train = self.container.get_service(
                    'io_manager_train')
            return mosaicking.Mosaicker(
                io_manager=io_manager_train,
                processor=self.container.get_service(
                    'processor_train', io_manager_train=io_manager_train),
                *args, **kwargs
            )
        self.container.register_service(
            'mosaicker_train',
            make_mosaicker_train,
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

        # Image processing
        def make_image_aligner_blender(
            image_transformer: raster.PassImageTransformer = None,
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

        # We'll include a scorer as well
        self.container.register_service(
            'image_scorer',
            scorers.SimilarityScorer,
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
