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

from . import preprocess, mosaic


class MosaicPipelines:

    @staticmethod
    def referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] =  'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('geotiff', preprocess.GeoTIFFPreprocesser(crs=crs)),
            ('geobounds', preprocess.GeoBoundsPreprocesser(
                crs=crs, passthrough=['filepath']
            )),
            ('mosaic', mosaic.ReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline
