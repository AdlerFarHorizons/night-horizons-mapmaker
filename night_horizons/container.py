'''This module contains the Dependency Injection Container class, which is
responsible for constructing the objects we use.
'''

from collections import OrderedDict
import inspect

import pyproj
from ruamel.yaml import YAML
from sklearn.utils import check_random_state

from .utils import deep_merge


class DIContainer:
    '''Dependency Injection Container. This class is responsible for
    constructing the objects we use. This has the benefit of a) enabling easy
    integration with a config, b) decoupling the object creation from the rest
    of the code (which e.g. allows us to update only one place when updating
    how objects are created), and c) decouples classes from each other.

    Jargon
    ------
    Container: This class.
    Service: An object that is created by the container.
    Dependency: An object that a service depends on.
    Inject: The process of providing a service with its dependencies.

    '''

    def __init__(self, config_filepath: str, local_options: dict = None):
        '''
        Initialize the DIContainer.

        Parameters
        ----------
        config_filepath : str
            The filepath to the configuration file.
        local_options : dict, optional
            Local options to override the configuration, by default {}.
        '''

        if local_options is None:
            local_options = {}

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
        name: str,
        constructor: callable,
        singleton: bool = False,
        wrapped_constructor = None,
    ):
        '''
        Register a service (object) with the container. Doing so allows us to 
        get an instance of it on-demand.

        Parameters
        ----------
        name : str
            The name of the service.
        constructor : callable
            The constructor function for the service.
        singleton : bool, optional
            Flag indicating if the service should be a singleton,
            by default False.
        wrapped_constructor : callable, optional
            The wrapped constructor function for the service,
            by default None.

        '''

        self._services[name] = {
            'constructor': constructor,
            'singleton': singleton,
            'wrapped_constructor': (
                wrapped_constructor if wrapped_constructor is not None
                else constructor
            ),
        }

    def get_service(self, name: str, *args, version: str = None, **kwargs):
        '''
        Get an instance of a registered service.

        Parameters
        ----------
        name : str
            The name of the service.
        version : str, optional
            The version of the service, by default None.
        *args
            Positional arguments to be passed to the service constructor.
        **kwargs
            Keyword arguments to be passed to the service constructor.

        Returns
        -------
        object
            An instance of the requested service.

        Raises
        ------
        ValueError
            If the service is not registered.

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
        args, kwargs = self.get_used_args(name, constructor, *args, **kwargs)

        # Finally, construct the service
        service = constructor(*args, **kwargs)
        if constructor_dict['singleton']:
            self.services[name] = service

        return service

    def get_used_args(
        self,
        name: str,
        constructor: callable,
        *args, **kwargs
    ) -> tuple[tuple, dict]:
        '''
        Get the used arguments for a service, based on the config,
        the arguments passed in, and the function defaults.

        Parameters
        ----------
        name : str
            The name of the service.
        constructor : callable
            The constructor function for the service.
        *args
            Positional arguments to be passed to the service constructor.
        **kwargs
            Keyword arguments to be passed to the service constructor.

        Returns
        -------
        tuple
            A tuple containing the positional arguments and keyword arguments
            to be used for constructing the service.

        '''

        # Start by combining the passed-in kwargs and the config
        if name in self.config:
            kwargs = deep_merge(self.config[name], kwargs)

        # Then fall back to the defaults
        default_kwargs = self.get_default_args(constructor)
        kwargs = deep_merge(default_kwargs, kwargs)

        return args, kwargs

    def get_default_args(self, constructor: callable) -> dict:
        '''
        Get the default arguments for a service constructor.

        Parameters
        ----------
        constructor : callable
            The constructor function for the service.

        Returns
        -------
        dict
            A dictionary containing the default arguments for the constructor.

        '''

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
        '''
        Parse the configuration and perform custom adjustments for
        some parameters.

        Parameters
        ----------
        config : dict
            The configuration dictionary.

        Returns
        -------
        dict
            The parsed configuration dictionary.

        '''

        def deep_interpret(unparsed: dict) -> dict:
            '''
            Interpret the configuration recursively.

            Parameters
            ----------
                unparsed : dict
                The unparsed configuration dictionary.

            Returns
            -------
                dict
                The parsed configuration dictionary.
            '''
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
        '''
        Save the configuration to a file.

        Parameters
        ----------
        filepath : str
            The filepath to save the configuration.

        '''

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
            _, kwargs = self.get_used_args(name, constructor)

            if kwargs != {}:
                doc[name] = kwargs

        with open(filepath, 'w', encoding='UTF-8') as file:
            yaml.dump(doc, file)
