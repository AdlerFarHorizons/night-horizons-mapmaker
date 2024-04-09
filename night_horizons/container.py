from collections import OrderedDict
import copy
import os
import inspect

import cv2
import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.utils import check_random_state
from ruamel.yaml import YAML

from .transformers import preprocessors
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import data_io
from .utils import get_method_parameters, deep_merge


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
        self._services = OrderedDict()
        # Publicly accessible services (instantiated)
        self.services = {}

        # Load the config
        yaml = YAML()
        with open(config_filepath, 'r', encoding='UTF-8') as file:
            self.config = yaml.load(file)

        self.config = deep_merge(self.config, local_options)

        self.config = self.parse_config(self.config)

    def register_service(
        self,
        name,
        constructor,
        singleton: bool = False,
        wrapped_constructor=None,
    ):
        # TODO: Allow users to specify the constructor? ChatGPT has suggestions
        #      for this via importlib. What we really want is to allow users
        #      to be able to specify e.g. FH135 vs FH145, and as simply as
        #      possible. Specifying constructors is cool, but probably not
        #      the right solution. The right solution might involve
        #      separating the service creation logic and the choice of how
        #      to analyze the data.
        self._services[name] = {
            'constructor': constructor,
            'singleton': singleton,
            'wrapped_constructor': (
                wrapped_constructor if wrapped_constructor is not None
                else constructor
            ),
        }

    def get_service(self, name, version=None, *args, **kwargs):
        '''
        TODO: Add parameter validation.
        '''

        # If we have a particular version of the service we want to use
        # we can specify it here.
        if version is None:
            # We need to check if the version is specified in the config
            config_version = self.config.get(name, {}).get('version', None)
            name = config_version if config_version is not None else name
        else:
            name = version

        # First, get ingredients for constructing the service
        constructor_dict = self._services.get(name)
        if not constructor_dict:
            raise ValueError(f'Service {name} not registered')

        # Parse constructor ingredients
        if constructor_dict['singleton'] and name in self.services:
            return self.services[name]
        constructor = constructor_dict['constructor']

        # Next, get the kwargs for the service
        # Start by combinining the passed-in kwargs and the config
        if name in self.config:
            kwargs = deep_merge(self.config[name], kwargs)

        # Then fall back to the defaults
        default_kwargs = self.get_arg_defaults(constructor)
        kwargs = deep_merge(default_kwargs, kwargs)

        # Finally, construct the service
        service = constructor(*args, **kwargs)
        if constructor_dict['singleton']:
            self.services[name] = service

        return service

    def get_arg_defaults(self, constructor):

        kwargs = {}
        try:
            signature = inspect.signature(constructor)
            signature_found = True
        except ValueError:
            # We use a boolean here (instead of placing the logic in the try
            # statement) to avoid catching other TypeErrors
            signature_found = False

        if signature_found:
            for key, value in signature.parameters.items():
                if value.default is not inspect.Parameter.empty:
                    kwargs[key] = value.default

        return kwargs

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

    def save_config(self, filepath: str):

        # Get the parameters for each service
        yaml = YAML()
        doc = yaml.map()
        for name, constructor_dict in self._services.items():
            constructor = constructor_dict['constructor']
            if constructor is None:
                doc[name] = self.config[name]
                continue

            wrapped_constructor = constructor_dict['wrapped_constructor']
            if wrapped_constructor is not None:
                constructor = constructor

            # Comment the name of the constructor
            doc.yaml_add_eol_comment(
                f'{constructor.__module__}.{constructor.__name__}',
                key=name,
            )

            # Get the used arguments
            # TODO: Need to change this back from getting just defaults
            kwargs = self.get_arg_defaults(constructor)

            if kwargs != {}:
                doc[name] = kwargs

        with open(filepath, 'w', encoding='UTF-8') as file:
            yaml.dump(doc, file)

    def register_dataio_services(self):

        # Register data io services
        self.dataio_services = []
        for subclass in data_io.DataIO.__subclasses__():
            key = subclass.name + '_io'
            self.register_service(key, subclass)
            self.dataio_services.append(key)

        return self.dataio_services
