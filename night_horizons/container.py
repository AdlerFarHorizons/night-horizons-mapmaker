import os

import numpy as np
import pandas as pd
import scipy
from sklearn.utils import check_random_state
import yaml
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import io_management, preprocessors
from .image_processing import base, mosaicking, processors


class DIContainer:

    def __init__(self, config_filepath: str, local_options: dict = {}):
        '''

        TODO: Rename "service" (and maybe container) to something more
        recognizable to scientists?

        Parameters
        ----------
        Returns
        -------
        '''

        self._services = {}

        # Load the config
        with open(config_filepath, 'r', encoding='UTF-8') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.update_config(local_options)

        self.parse_config()

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

    def update_config(self, new_config):

        def deep_update(orig_dict, new_dict):
            for key, value in new_dict.items():
                if (
                    isinstance(value, dict)
                    and (key in orig_dict)
                    and isinstance(orig_dict[key], dict)
                ):
                    deep_update(orig_dict[key], value)
                else:
                    orig_dict[key] = value

        deep_update(self.config, new_config)

    def parse_config(self):
        '''This goes through the config and handles some parameters.

        Parameters
        ----------
        Returns
        -------
        '''

        for key, value in self.config['filetree'].items():
            self.config['filetree'][key] = os.path.join(
                self.config['root_dir'], value)

        if 'random_state' in self.config:
            self.config['random_state'] = check_random_state(
                self.config['random_state'])


class MosaickerFactory(DIContainer):

    def __init__(self, config_filepath: str, local_options: dict = {}):

        super().__init__(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        # We register the preprocessing here, in addition to the other objects
        self.register_service(
            'preprocessor',
            preprocessors.GeoTIFFPreprocessor
        )

        # Register file manager typical for mosaickers
        self.register_service(
            'io_manager',
            io_management.MosaicIOManager,
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
            io_manager: io_management.IOManager = None,
            row_processor: base.BaseRowProcessor = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service(
                    'io_manager',
                    out_dir=out_dir,
                )
            if row_processor is None:
                row_processor = self.get_service('row_processor')
            return mosaicking.BaseMosaicker(
                io_manager=io_manager,
                row_processor=row_processor,
                *args, **kwargs
            )
        self.register_service('mosaicker', make_mosaicker)

    def create(self, *args, **kwargs):

        return self.get_service('mosaicker', *args, **kwargs)


class SequentialMosaickerFactory(DIContainer):

    def __init__(self, config_filepath: str, local_options: dict = {}):

        super().__init__(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        # We register the preprocessing here, in addition to the other objects
        self.register_service(
            'y_preprocessor',
            preprocessors.GeoTIFFPreprocessor
        )

        # Register file manager typical for mosaickers
        self.register_service(
            'io_manager',
            io_management.MosaicIOManager,
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
            io_manager: io_management.IOManager = None,
            row_processor: base.BaseRowProcessor = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service(
                    'io_manager',
                    out_dir=out_dir,
                )
            if row_processor is None:
                row_processor = self.get_service('row_processor')
            return mosaicking.BaseMosaicker(
                io_manager=io_manager,
                row_processor=row_processor,
                *args, **kwargs
            )
        self.register_service('mosaicker', make_mosaicker)

    def create(self, *args, **kwargs):

        return self.get_service('mosaicker', *args, **kwargs)