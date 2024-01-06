import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.container import DIContainer
from night_horizons.io_manager import IOManager
from night_horizons.image_processing.processors import DatasetRegistrar
from night_horizons.image_processing.operators import BaseImageOperator


# Register services
local_options = {
    'io_manager': {
        'output_dir': './test/test_data/temp',
        'output_description': {
            'referenced_images_dir': 'referenced_images'
        },
    },
}
container = DIContainer('./test/config.yml', local_options=local_options)
container.register_service('io_manager', IOManager)
container.register_service('image_operator', MagicMock(spec=BaseImageOperator))
container.register_service(
    'dataset_registrar',
    lambda use_safe_process=False, *args, **kwargs:
        DatasetRegistrar(
            io_manager=container.get_service('io_manager'),
            image_operator=container.get_service('image_operator'),
            use_safe_process=use_safe_process,
            *args, **kwargs
        ),
)


class TestDatasetRegistrar(unittest.TestCase):

    def test_save_image_as_dataset(self):

        processor = container.get_service('dataset_registrar')

        assert False
