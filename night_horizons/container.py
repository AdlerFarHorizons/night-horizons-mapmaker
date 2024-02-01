import os
import inspect

import cv2
import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.utils import check_random_state
import yaml

from .transformers import preprocessors
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import data_io
from .utils import get_method_parameters


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

        # Internal services (constructors)
        self._services = {}
        # Publicly accessible services (instantiated)
        self.services = {}

        # Load the config
        with open(config_filepath, 'r', encoding='UTF-8') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.config = self.update_config(self.config, local_options)

        self.config = self.parse_config(self.config)

    def register_service(
        self,
        name,
        constructor,
        singleton: bool = False,
        args_key: str = None,
    ):
        self._services[name] = {
            'constructor': constructor,
            'singleton': singleton,
            'args_key': args_key,
        }

    def get_service(self, name, *args, **kwargs):
        '''
        TODO: Add parameter validation.
        '''

        # Get parameters for constructing the service
        constructor_dict = self._services.get(name)
        if not constructor_dict:
            raise ValueError(f'Service {name} not registered')

        # Parse constructor parameters
        if constructor_dict['singleton'] and name in self.services:
            return self.services[name]
        constructor = constructor_dict['constructor']

        # Get the used arguments
        if constructor_dict['args_key'] is None:
            args_key = name
        else:
            args_key = constructor_dict['args_key']
        kwargs = self.get_service_args(args_key, constructor, **kwargs)

        return constructor(*args, **kwargs)

    def get_service_args(self, name, constructor, **kwargs):

        # Get config values
        if name in self.config:
            kwargs = {**self.config[name], **kwargs}

        try:
            signature = inspect.signature(constructor)
            signature_found = True
        except ValueError:
            # We use a boolean here (instead of placing the logic in the try
            # statement) to avoid catching other TypeErrors
            signature_found = False

        if signature_found:

            # TODO: Delete
            # # Access the global arguments
            # if 'global' in self.config:
            #     global_kwargs = {}
            #     for key in signature.parameters.keys():
            #         if key in self.config['global']:
            #             global_kwargs[key] = self.config['global'][key]
            #     kwargs = {**global_kwargs, **kwargs}

            # Advanced: when the values are dictionaries, blend them
            #           this is important for input and output descriptions
            # TODO: This currently only works for defaults, but we should
            #       also be able to blend dictionaries for config vs passed-in
            for key, value in kwargs.items():
                if key not in signature.parameters.keys():
                    continue
                default_value = signature.parameters[key].default
                if isinstance(value, dict) and isinstance(default_value, dict):
                    kwargs[key] = {**default_value, **value}

        return kwargs

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
        self.dataio_services = []
        for subclass in data_io.DataIO.__subclasses__():
            key = subclass.name + '_io'
            self.register_service(key, subclass)
            self.dataio_services.append(key)

        return self.dataio_services
