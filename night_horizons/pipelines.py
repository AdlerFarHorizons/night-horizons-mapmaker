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


class GeoreferencePipelines:

    @staticmethod
    def sensor_georeference(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocess.NITELitePreprocesser(
                output_columns=['sensor_x', 'sensor_y'])),
            ('sensor_georeference', reference.SensorGeoreferencer(crs=crs)),
        ])

        y_pipeline = preprocess.GeoTIFFPreprocesser(crs=crs)

        return pipeline, y_pipeline


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
