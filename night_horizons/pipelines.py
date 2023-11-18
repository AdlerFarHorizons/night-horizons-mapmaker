'''Pipelines for various purposes.
'''

from typing import Union

import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import preprocess, reference, mosaic


class PreprocessingPipelines:

    @staticmethod
    def get_metadata_and_approx_georefs(
        passthrough: list[str] = ['filepath', 'camera_num'],
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocess.NITELitePreprocesser(crs=crs)),
            ('georeference', ColumnTransformer(
                transformers=[
                    ('georeference', reference.SensorGeoreferencer(crs=crs),
                     ['sensor_x', 'sensor_y']),
                    ('passthrough', 'passthrough', passthrough)
                ],
                remainder='drop',
            )),
        ])

        return pipeline


class MosaicPipelines:

    @staticmethod
    def referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('geotiff', preprocess.GeoTIFFPreprocesser(crs=crs)),
            ('mosaic', mosaic.ReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline

    @staticmethod
    def less_referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocess.NITELitePreprocesser(
                output_columns=['filepath', 'sensor_x', 'sensor_y'])),
            ('geotiff', preprocess.GeoTIFFPreprocesser(
                crs=crs, passthrough=True)),
            ('mosaic', mosaic.LessReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline
