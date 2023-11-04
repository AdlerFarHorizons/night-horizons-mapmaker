'''Pipelines for various purposes.
'''

from typing import Union

import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator
import sklearn.pipeline as sk_pipeline
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import preprocess, reference


class PreprocessingPipelines:

    @staticmethod
    def referenced_nitelite(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        output_columns: list[str] = ['filepath', 'sensor_x', 'sensor_y'],
    ):

        pipeline = sk_pipeline.Pipeline([
            (
                'nitelite',
                preprocess.NITELitePreprocesser(
                    # We choose these columns since they're the ones
                    # needed for GeoTIFF preprocessing
                    output_columns=output_columns,
                    crs=crs,
                )
            ),
            (
                'geotiff',
                preprocess.GeoTIFFPreprocesser(crs=crs, passthrough=True),
            ),
        ])

        return pipeline


class GeoreferencingPipelines:

    @staticmethod
    def sensor_georeferencing(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        preprocessing: BaseEstimator =
            PreprocessingPipelines.referenced_nitelite(),
    ):

        pipeline = sk_pipeline.Pipeline([
            ('preprocessing', preprocessing),
            ('sensor_georeferencing', reference.SensorGeoreferencer(crs=crs)),
        ])

        return pipeline
