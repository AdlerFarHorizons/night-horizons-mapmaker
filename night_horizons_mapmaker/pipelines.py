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

from . import preprocess, reference


class PreprocessingPipelines:

    @staticmethod
    def referenced_nitelite(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        output_columns: list[str] = ['filepath', 'sensor_x', 'sensor_y', ]
    ):

        pipeline = sk_pipeline.Pipeline([
            (
                'nitelite_preprocessing',
                preprocess.NITELitePreprocesser(
                    output_columns=output_columns,
                    crs=crs,
                )
            ),
            ('geo_preprocessing', preprocess.GeoTIFFPreprocesser(crs=crs)),
        ])

        return pipeline


class GeoreferencingPipelines:

    @staticmethod
    def sensor_georeferencing(crs: Union[str, pyproj.CRS] = 'EPSG:3857'):

        pipeline = sk_pipeline.Pipeline([
            (
                'metadata_preprocessing',
                preprocess.NITELitePreprocesser(
                    output_columns=['filepath', 'sensor_x', 'sensor_y', ],
                    crs=crs,
                )
            ),
            ('geo_preprocessing', preprocess.GeoTIFFPreprocesser(crs=crs)),
            ('sensor_georeferencing', reference.SensorGeoreferencer(crs=crs)),
        ])

        return pipeline
