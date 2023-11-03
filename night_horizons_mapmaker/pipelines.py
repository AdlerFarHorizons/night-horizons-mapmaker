'''Pipelines for various purposes.
'''

from typing import Union

import numpy as np
import pandas as pd
import pyproj
import scipy
import sklearn.pipeline as sk_pipeline
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import preprocess


class GeoreferencingPipelines:

    @staticmethod
    def sensor_georeferencing(crs: Union[str, pyproj.CRS] = 'EPSG:3857'):

        pipeline = sk_pipeline.Pipeline([
            (
                'metadata_preprocessing',
                preprocess.MetadataPreprocesser(
                    output_columns=['filepath', 'sensor_x', 'sensor_y', ],
                    crs=crs,
                )
            ),
            ('geo_preprocessing', preprocess.GeoPreprocesser(crs=crs)),
        ])
