
from .container import DIContainer
from . import io_manager, preprocessors
from .image_processing import mosaicking, operators, processors


class Mapmaker:
    def __init__(self, config_filepath: str, local_options: dict = {}):
        self.container = DIContainer(
            config_filepath=config_filepath,
            local_options=local_options,
        )

    def run(self):
        # Preprocessing
        preprocessor = self.container.create_preprocessor(self.config)
        preprocessed_data = preprocessor.preprocess()

        # Batch processing
        batch_processor = self.container.create_batch_processor(self.config)
        processed_data = batch_processor.process(preprocessed_data)

        # Postprocessing
        postprocessor = self.container.create_postprocessor(self.config)
        postprocessed_data = postprocessor.postprocess(processed_data)

        # Additional logic or output here


class MosaicMaker(Mapmaker):

    def __init__(self, config_filepath: str, local_options: dict = {}):

        super().__init__(
            config_filepath=config_filepath,
            local_options=local_options,
        )

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
        def make_mosaicker_processor(
            io_manager: io_manager.IOManager = None,
            image_operator: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service('io_manager')
            if image_operator is None:
                image_operator = self.get_service('image_operator')
            return processors.Processor(
                io_manager=io_manager,
                image_operator=image_operator,
                *args, **kwargs
            )
        self.container.register_service('processor', make_mosaicker_processor)

        # Finally, the mosaicker itself, which is a batch processor
        def make_mosaicker(
            io_manager: io_manager.IOManager = None,
            processor: processors.Processor = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service('io_manager')
            if processor is None:
                processor = self.get_service(
                    'processor',
                    io_manager=io_manager,
                )
            return mosaicking.Mosaicker(
                io_manager=io_manager,
                processor=processor,
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
