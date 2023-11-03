'''Main module for performing georeferencing
'''
from typing import Union

import numpy as np
import pyproj
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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
        self.is_fitted_ = True

        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Calculate offsets
        widths = X['pixel_width'] * X['n_x']
        heights = X['pixel_height'] * X['n_y']
        x_centers = X['x_min'] + 0.5 * widths
        y_centers = X['y_max'] + 0.5 * heights
        offsets = np.sqrt(
            (X['sensor_x'] - x_centers)**2.
            + (X['sensor_y'] - y_centers)**2.
        )

        # Estimate values that are just averages
        self.estimated_width_ = np.nanmedian(widths)
        self.estimated_height_ = np.nanmedian(heights)
        self.estimated_pixel_width_ = np.nanmedian(X['pixel_width'])
        self.estimated_pixel_height_ = np.nanmedian(X['pixel_height'])
        self.estimated_nx_ = np.round(
            self.estimated_width_ / self.estimated_pixel_width_
        ).astype(int)
        self.estimated_ny_ = np.round(
            self.estimated_height_ / self.estimated_pixel_height_
        ).astype(int)
        self.estimated_spatial_offset_ = np.nanpercentile(
            offsets,
            self.q_offset
        )

        # `fit` should always return `self`
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

        return np.ones(X.shape[0], dtype=np.int64)