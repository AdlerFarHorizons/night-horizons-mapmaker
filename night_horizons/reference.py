'''Main module for performing georeferencing
'''
from typing import Union

import numpy as np
import pyproj
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import utils, preprocess


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
        passthrough: Union[bool, list[str]] = False,
    ):
        self.crs = crs
        self.passthrough = passthrough
        self.required_columns = ['sensor_x', 'sensor_y']

    @utils.enable_passthrough
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
        utils.check_df_input(
            X,
            self.required_columns,
        )

        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Calculate offsets
        widths = y['pixel_width'] * y['x_size']
        heights = -y['pixel_height'] * y['y_size']

        # Estimate values that are just averages
        self.width_ = np.nanmedian(widths)
        self.height_ = np.nanmedian(heights)
        self.pixel_width_ = np.nanmedian(y['pixel_width'])
        self.pixel_height_ = np.nanmedian(y['pixel_height'])
        self.x_rot_ = np.nanmedian(y['x_rot'])
        self.y_rot_ = np.nanmedian(y['y_rot'])
        self.x_size_ = np.round(
            self.width_ / self.pixel_width_
        ).astype(int)
        self.y_size_ = np.round(np.abs(
            self.height_ / self.pixel_height_
        )).astype(int)

        # `fit` should always return `self`
        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
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

        check_is_fitted(self, 'is_fitted_')
        utils.check_df_input(
            X,
            self.required_columns,
        )

        # Calculate properties
        corners_and_centers = [
            'x_min', 'x_max',
            'y_min', 'y_max',
            'x_center', 'y_center',
        ]
        for key in preprocess.GEOTRANSFORM_COLS:
            if key in corners_and_centers:
                continue
            X[key] = getattr(self, key + '_')
        X['x_min'] = X['sensor_x'] - 0.5 * self.width_
        X['x_max'] = X['sensor_x'] + 0.5 * self.width_
        X['y_min'] = X['sensor_y'] - 0.5 * self.height_
        X['y_max'] = X['sensor_y'] + 0.5 * self.height_
        X['x_center'] = X['sensor_x']
        X['y_center'] = X['sensor_y']

        return X

    def score_samples(self, X, y):

        check_is_fitted(self, 'is_fitted_')

        y_pred = self.predict(X)
        pred_xs = 0.5 * (y_pred['x_min'] + y_pred['x_max'])
        pred_ys = 0.5 * (y_pred['y_min'] + y_pred['y_max'])

        actual_xs = 0.5 * (y['x_min'] + y['x_max'])
        actual_ys = 0.5 * (y['y_min'] + y['y_max'])

        self.scores_ = np.sqrt(
            (pred_xs - actual_xs)**2.
            + (pred_ys - actual_ys)**2.
        )

        return self.scores_

    def score(self, X, y):

        return np.median(self.score_samples(X, y))
