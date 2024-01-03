
from .container import DIContainer
from . import io_manager, preprocessors
from .image_processing import mosaicking, operators, processors


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
        def make_mosaicker_processor(
            io_manager: io_manager.IOManager = None,
            image_operator: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.container.get_service('io_manager')
            if image_operator is None:
                image_operator = self.container.get_service('image_operator')
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
                io_manager = self.container.get_service('io_manager')
            if processor is None:
                processor = self.container.get_service(
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
