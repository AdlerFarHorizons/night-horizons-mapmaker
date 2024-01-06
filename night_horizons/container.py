import os
import inspect

import cv2
import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.utils import check_random_state
import yaml
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import data_io, io_manager, pipelines, preprocessors
from .image_processing import mosaicking, operators, processors


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

        self.config = self.update_config(self.config, local_options)

        self.config = self.parse_config(self.config)

    def register_service(self, name, constructor):
        self._services[name] = constructor

    def get_service(self, name, *args, **kwargs):
        '''
        TODO: Add parameter validation.
        '''

        constructor = self._services.get(name)
        if not constructor:
            raise ValueError(f'Service {name} not registered')

        # Get the global arguments
        callargs = inspect.getargspec(constructor).args
        global_kwargs = {}
        for arg in callargs:
            if arg in self.config['global']:
                global_kwargs[arg] = self.config[arg]
        kwargs = {**global_kwargs, **kwargs}

        # The used kwargs are a merger of the config values and those passed in
        if name in self.config:
            kwargs = {**self.config[name], **kwargs}

        return constructor(*args, **kwargs)

    def update_config(self, old_config, new_config):

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

        deep_update(old_config, new_config)

        return old_config

    def parse_config(self, config: dict) -> dict:
        '''This goes through the config and handles some parameters.

        Parameters
        ----------
        Returns
        -------

        '''

        # TODO: Delete this once we're sure we don't need it
        # for key, value in config['filetree'].items():
        #     config['filetree'][key] = os.path.join(config['root_dir'], value)
        # config.setdefault('io_manager', {})['out_dir'] = \
        #     config['filetree']['out_dir']

        def deep_interpret(unparsed):

            parsed = {}
            for key, value in unparsed.items():

                if isinstance(value, dict):
                    parsed[key] = deep_interpret(value)
                elif key == 'random_state':
                    parsed[key] = check_random_state(value)
                elif key == 'crs':
                    if not isinstance(value, pyproj.CRS):
                        parsed[key] = pyproj.CRS(value)
                    else:
                        parsed[key] = value
                else:
                    parsed[key] = value

            return parsed

        return deep_interpret(config)

    def register_dataio_services(self):

        # Register data io services
        for subclass in data_io.DataIO.__subclasses__():
            self.register_service(subclass.name, subclass)
