'''Main module for performing georeferencing
'''
from typing import Union

import numpy as np
import pyproj
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import preprocess


class SensorGeoreferencer(BaseEstimator):
    '''Perform georeferencing based on sensor metadata.

    Parameters
    ----------
    crs:
        Coordinate reference system to use.
    '''
    def __init__(
        self,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        q_offset: float = 0.95,
    ):
        self.crs = crs
        self.q_offset = q_offset

    def fit(self, X, y):
        '''A reference implementation of a fitting function.

        TODO: This currently parameterizes the estimates using the
            geotransforms, which are anchored by x_min and y_max.
            However, the sensor x and y are more-closely related to
            x_center and y_center. Parameterizing by those would make
            more sense, but would require tracking the differences.

        Parameters
        ----------
        X :
            Sensor coords and filepaths.
        y :
            Geotransforms identifying location.

        Returns
        -------
        self : object
            Returns self.
        '''
        X, y = check_X_y(X, y, multi_output=True)

        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Calculate offsets
        widths = y['pixel_width'] * y['n_x']
        heights = y['pixel_height'] * y['n_y']
        x_centers = y['x_min'] + 0.5 * widths
        y_centers = y['y_max'] + 0.5 * heights
        offsets = np.sqrt(
            (X['sensor_x'] - x_centers)**2.
            + (X['sensor_y'] - y_centers)**2.
        )

        # Estimate values that are just averages
        self.width_ = np.nanmedian(widths)
        self.height_ = np.nanmedian(heights)
        self.pixel_width_ = np.nanmedian(y['pixel_width'])
        self.pixel_height_ = np.nanmedian(y['pixel_height'])
        self.nx_ = np.round(
            self.width_ / self.pixel_width_
        ).astype(int)
        self.ny_ = np.round(
            self.height_ / self.pixel_height_
        ).astype(int)
        self.spatial_offset_ = np.nanpercentile(
            offsets,
            self.q_offset
        )

        # `fit` should always return `self`
        self.is_fitted_ = True
        return self

    def predict(self, X):
        ''' A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        '''
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # Calculate properties
        for key in preprocess.GEOTRANSFORM_COLS:
            if key in ['x_min', 'y_max']:
                continue
            X[key] = getattr(self, key + '_')
        X['x_min'] = X['sensor_x'] - 0.5 * self.width_
        X['y_max'] = X['sensor_y'] - 0.5 * self.height_

        return X
