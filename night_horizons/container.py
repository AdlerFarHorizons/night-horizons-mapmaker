import numpy as np
import pandas as pd
import scipy
import yaml
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import file_management, preprocessors
from .image_processing import base, mosaicking, processors


class DIContainer:

    def __init__(self, config_filepath: str):
        '''

        TODO: Rename "service" (and maybe container) to something more
        recognizable to scientists.

        Parameters
        ----------
        Returns
        -------
        '''

        self._services = {}

        # Load the config
        with open(config_filepath, 'r', encoding='UTF-8') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def register_service(self, name, constructor):
        self._services[name] = constructor

    def get_service(self, name, *args, **kwargs):
        constructor = self._services.get(name)
        if not constructor:
            raise ValueError(f'Service {name} not registered')

        # The used kwargs are a merger of the config values and those passed in
        if name in self.config:
            kwargs = {**self.config[name], **kwargs}
        return constructor(*args, **kwargs)


class MosaickerMaker(DIContainer):

    def __init__(self, config_filepath: str):

        super().__init__(config_filepath=config_filepath)

        # We register the preprocessing here too
        self.register_service(
            'preprocessing',
            preprocessors.GeoTIFFPreprocessor
        )

        # Register file manager typical for mosaickers
        self.register_service(
            'file_manager',
            file_management.MosaicFileManager,
        )

        # Image processor typical for mosaickers (constructor defaults are ok)
        self.register_service(
            'image_blender',
            processors.ImageBlender,
        )

        # And the row transformer typical for mosaickers
        def make_mosaicker_row_processor(
            image_processor: processors.ImageProcessor = None,
            *args, **kwargs
        ):
            if image_processor is None:
                image_processor = self.get_service('image_blender')
            return mosaicking.MosaickerRowTransformer(
                image_processor=image_processor,
                *args, **kwargs
            )
        self.register_service(
            'row_processor',
            make_mosaicker_row_processor,
        )

        # Finally, the mosaicker itself
        def make_mosaicker(
            out_dir: str,
            file_manager: file_management.FileManager = None,
            row_processor: base.BaseRowProcessor = None,
            *args, **kwargs
        ):
            if file_manager is None:
                file_manager = self.get_service(
                    'file_manager',
                    out_dir=out_dir,
                )
            if row_processor is None:
                row_processor = self.get_service('row_processor')
            return mosaicking.BaseMosaicker(
                file_manager=file_manager,
                row_processor=row_processor,
                *args, **kwargs
            )
        self.register_service('mosaicker', make_mosaicker)

    def create(self, *args, **kwargs):

        return self.get_service('mosaicker', *args, **kwargs)