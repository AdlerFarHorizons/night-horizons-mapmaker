import numpy as np
import pandas as pd
import scipy
import yaml
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from .image_processing import mosaicking, processors


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


class MosaickerManager(DIContainer):

    def __init__(self, config_filepath: str):

        super().__init__(config_filepath=config_filepath)

        self.register_service(
            'image_blender',
            processors.ImageBlender,
        )
        self.register_service(
            'row_transformer',
            lambda *args, **kwargs: mosaicking.MosaickerRowTransformer(
                image_processor=self.get_service('image_blender'),
                *args, **kwargs
            ),
        )
        self.register_service(
            'mosaicker',
            lambda *args, **kwargs: mosaicking.BaseMosaicker(
                row_processor=self.get_service('row_transformer'),
                *args, **kwargs
            )
        )
