import os

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

from . import io, pipelines, preprocessors
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

        for key, value in config['filetree'].items():
            config['filetree'][key] = os.path.join(config['root_dir'], value)
        config.setdefault('io_manager', {})['out_dir'] = \
            config['filetree']['out_dir']

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
            io.MosaicIOManager,
        )

        # Image processor typical for mosaickers (constructor defaults are ok)
        self.register_service(
            'image_blender',
            operators.ImageBlender,
        )

        # And the row transformer typical for mosaickers
        def make_mosaicker_row_processor(
            image_processor: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if image_processor is None:
                image_processor = self.get_service('image_blender')
            return processors.MosaickerRowTransformer(
                image_processor=image_processor,
                *args, **kwargs
            )
        self.register_service(
            'row_processor',
            make_mosaicker_row_processor,
        )

        # Finally, the mosaicker itself
        def make_mosaicker(
            io_manager: io.OutputFileManager = None,
            row_processor: processors.Processor = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service('io_manager')
            if row_processor is None:
                row_processor = self.get_service('row_processor')
            return mosaicking.Mosaicker(
                io_manager=io_manager,
                row_processor=row_processor,
                *args, **kwargs
            )
        self.register_service('mosaicker', make_mosaicker)

    def create(self, *args, **kwargs):

        return self.get_service('mosaicker', *args, **kwargs)


class SequentialMosaickerFactory(DIContainer):
    '''TODO: Can we clean this up? Possibly incorporate this into a Mapmaker
             class?
    '''

    def __init__(self, config_filepath: str, local_options: dict = {}):

        super().__init__(
            config_filepath=config_filepath,
            local_options=local_options,
        )

        # We register the preprocessing here, in addition to the other objects
        # Preprocessor for X values
        self.register_service(
            'preprocessor',
            pipelines.PreprocessorPipelines.nitelite
        )
        # Preprocessor for y values
        self.register_service(
            'preprocessor_y',
            preprocessors.GeoTIFFPreprocessor
        )

        # Register file manager typical for mosaickers
        self.register_service(
            'io_manager',
            io.MosaicIOManager,
        )

        # Feature detection and matching
        self.register_service(
            'image_transformer',
            preprocessors.PassImageTransformer,
        )
        self.register_service(
            'feature_detector',
            cv2.AKAZE.create,
        )
        self.register_service(
            'feature_matcher',
            cv2.BFMatcher.create,
        )

        # Image processing
        def make_image_aligner_blender(
            image_transformer: preprocessors.PassImageTransformer = None,
            feature_detector: cv2.Feature2D = None,
            feature_matcher: cv2.DescriptorMatcher = None,
            *args, **kwargs
        ):
            if image_transformer is None:
                image_transformer = self.get_service('image_transformer')
            if feature_detector is None:
                feature_detector = self.get_service('feature_detector')
            if feature_matcher is None:
                feature_matcher = self.get_service('feature_matcher')
            return operators.ImageAlignerBlender(
                image_transformer=image_transformer,
                feature_detector=feature_detector,
                feature_matcher=feature_matcher,
                *args, **kwargs
            )
        self.register_service(
            'image_processor',
            make_image_aligner_blender,
        )
        # For the training mosaic
        self.register_service(
            'image_blender',
            operators.ImageBlender,
        )

        # And the row transformer typical for mosaickers
        def make_mosaicker_row_processor_train(
            image_processor: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if image_processor is None:
                image_processor = self.get_service('image_blender')
            return processors.MosaickerRowTransformer(
                image_processor=image_processor,
                *args, **kwargs
            )
        self.register_service(
            'row_processor_train',
            make_mosaicker_row_processor_train,
        )

        # And the row transformer used for the sequential mosaicker
        def make_mosaicker_row_processor(
            image_processor: operators.BaseImageOperator = None,
            *args, **kwargs
        ):
            if image_processor is None:
                image_processor = self.get_service('image_processor')
            return processors.SequentialMosaickerRowTransformer(
                image_processor=image_processor,
                *args, **kwargs
            )
        self.register_service(
            'row_processor',
            make_mosaicker_row_processor
        )

        def make_mosaicker_train(
            io_manager_train: io.OutputFileManager = None,
            row_processor_train: processors.Processor = None,
            *args, **kwargs
        ):
            if io_manager_train is None:
                io_manager_train = self.get_service(
                    'io_manager',
                    file_exists='pass',
                    aux_files={'settings': 'settings_initial.yaml'},
                )
            if row_processor_train is None:
                row_processor_train = self.get_service('row_processor_train')
            return mosaicking.Mosaicker(
                io_manager=io_manager_train,
                row_processor=row_processor_train,
                *args, **kwargs
            )
        self.register_service('mosaicker_train', make_mosaicker_train)

        # Finally, the mosaicker itself
        def make_mosaicker(
            io_manager: io.OutputFileManager = None,
            row_processor: processors.Processor = None,
            mosaicker_train: mosaicking.Mosaicker = None,
            *args, **kwargs
        ):
            if io_manager is None:
                io_manager = self.get_service('io_manager')
            if row_processor is None:
                row_processor = self.get_service('row_processor')
            if mosaicker_train is None:
                mosaicker_train = self.get_service('mosaicker_train')
            return mosaicking.SequentialMosaicker(
                io_manager=io_manager,
                row_processor=row_processor,
                mosaicker_train=mosaicker_train,
                *args, **kwargs
            )
        self.register_service('mosaicker', make_mosaicker)

    def create(self, *args, **kwargs):

        return self.get_service('mosaicker', *args, **kwargs)
