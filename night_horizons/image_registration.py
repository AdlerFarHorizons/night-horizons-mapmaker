'''Main module for performing georeferencing
'''
from typing import Union

import numpy as np
import pandas as pd
import pyproj
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import preprocessers, utils


class MetadataImageRegistrar(BaseEstimator):
    '''Perform georeferencing based on sensor metadata.

    Parameters
    ----------
    crs:
        Coordinate reference system to use.
    '''
    def __init__(
        self,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        passthrough: Union[bool, list[str]] = False,  # TODO: Deprecated.
        use_direct_estimate: bool = True,
        camera_angles: dict[float] = {0: 30., 1: 0., 2: 30.},
        angle_error: float = 5.,
        padding_fraction: float = 1.0,
    ):
        self.crs = crs
        self.passthrough = passthrough
        self.required_columns = [
            'sensor_x',
            'sensor_y',
            'camera_num',
            'mAltitude',
        ]
        self.use_direct_estimate = use_direct_estimate
        self.camera_angles = camera_angles
        self.angle_error = angle_error
        self.padding_fraction = padding_fraction

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

        # Estimate spatial error
        X['offset'] = np.sqrt(
            (X['sensor_x'] - y['x_center'])**2.
            + (X['sensor_y'] - y['y_center'])**2.
        )
        X_cam = X.groupby('camera_num')
        self.spatial_error_ = X_cam['offset'].mean()

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
        set_manually = [
            'x_min', 'x_max',
            'y_min', 'y_max',
            'x_center', 'y_center',
            'spatial_error', 'padding',
        ]
        for key in preprocessers.GEOTRANSFORM_COLS:
            if key in set_manually:
                continue
            X[key] = getattr(self, key + '_')
        X['x_min'] = X['sensor_x'] - 0.5 * self.width_
        X['x_max'] = X['sensor_x'] + 0.5 * self.width_
        X['y_min'] = X['sensor_y'] - 0.5 * self.height_
        X['y_max'] = X['sensor_y'] + 0.5 * self.height_
        X['x_center'] = X['sensor_x']
        X['y_center'] = X['sensor_y']

        # Estimate spatial error
        X['spatial_error'] = np.nan
        # First, we identify what cameras we can use a direct inference for.
        if hasattr(self, 'spatial_error_') and self.use_direct_estimate:
            has_direct_estimate = X['camera_num'].isin(
                self.spatial_error_.index)
            X.loc[has_direct_estimate, 'spatial_error'] = \
                self.spatial_error_.loc[
                    X.loc[has_direct_estimate, 'camera_num']].values
        # Second, fall-back to expected values based on camera angles
        is_na = X['spatial_error'].isna()
        if is_na.sum() > 0:
            camera_angles = X['camera_num'].map(self.camera_angles)
            X.loc[is_na, 'spatial_error'] = X.loc[is_na, 'mAltitude'] * np.tan(
                (camera_angles + self.angle_error) * np.pi / 180.
            )

        # Add padding
        X['padding'] = self.padding_fraction * X['spatial_error']

        # Ensure correct type
        X[preprocessers.GEOTRANSFORM_COLS] = \
            X[preprocessers.GEOTRANSFORM_COLS].astype(float)

        return X

    def transform(self, X):
        return self.predict(X)

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
