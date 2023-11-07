import glob
import os
from typing import Union
import warnings

import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ReferencedMosaic(TransformerMixin, BaseEstimator):
    '''Assemble a mosaic from georeferenced images.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):
        self.crs = crs

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        img_log_fp: str = None,
        imu_log_fp: str = None,
        gps_log_fp: str = None,
    ):

        # Check the input is good.
        X = check_input(X)

        self.is_fitted_ = True
        return self


def check_input(
    X: pd.DataFrame,
    passthrough: bool = False,
) -> pd.DataFrame:
    '''Input check for acceptable types for preprocessing.

    Parameters
    ----------
        X:
            Input data.
    Returns
    -------
    '''

    assert isinstance(X, pd.DataFrame), (
        'X must be a dataframe with columns'
        '[filepath, x_min, x_max, y_min, y_max, n_x, n_y]'
    )

    if isinstance(X, pd.DataFrame):
        if not passthrough:
            assert X.columns == ['filepath'], (
                'Unexpected columns in preprocesser input.'
            )
        return X

    # We offer some minor reshaping to be compatible with common
    # expectations that a single list of features doesn't need to be 2D.
    if len(np.array(X).shape) == 1:
        X = np.array(X).reshape(1, len(X))

    # Check and unpack X
    X = check_array(X, dtype='str')
    X = pd.DataFrame(X.transpose(), columns=['filepath'])

    return X

