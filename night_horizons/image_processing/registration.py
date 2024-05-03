"""Main module for performing georeferencing. Only includes relatively simple methods
for performing image registration. The others are typically aligned as part of a batch
process.
"""

from typing import Union

import numpy as np
import pandas as pd
import pyproj
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..transformers import preprocessors

from .. import utils


class MetadataImageRegistrar(BaseEstimator):
    """Perform georeferencing based on sensor metadata.
    This gives an approximate search zone, but is not accurate enough to use for
    anything else.
    """

    def __init__(
        self,
        crs: pyproj.CRS,
        passthrough: Union[bool, list[str]] = False,
        use_observed_error: bool = True,
        camera_angles: dict[float] = {0: 30.0, 1: 0.0, 2: 30.0},
        angle_error: float = 5.0,
        padding_fraction: float = 1.5,
    ):
        """
        Initialize the MetadataImageRegistrar object.

        Parameters
        ----------
        crs : pyproj.CRS
            The coordinate reference system (CRS) to use for the registration,
            by default 'EPSG:3857'.
        passthrough : Union[bool, list[str]], optional
            Whether to pass through certain columns from the input DataFrame,
            by default False.
            If False, only the required columns will be kept.
            If True, all columns will be passed through.
            If a list of column names is provided, only those columns
            will be passed through.
        use_observed_error : bool, optional
            Whether to use the error in the sensor coordinates, when possible.
            Otherwise we estimate the error using the height and the camera angle.
        camera_angles : dict[float], optional
            The camera angles for each camera number,
            by default {0: 30., 1: 0., 2: 30.}.
            The keys are the camera numbers and the values are
            the corresponding angles in degrees.
        angle_error : float, optional
            The assumed error in a camera's pointing; part of the calculation when
            using the observed error is not possible.
        padding_fraction : float, optional
            The fraction of padding to add around the edges of the estimated
            registered image, in units of the hypotenuse of the estimated registered
            image. Defaults to 1.5.
        """
        self.crs = crs
        self.passthrough = passthrough
        self.required_columns = [
            "sensor_x",
            "sensor_y",
            "camera_num",
            "mAltitude",
        ]
        self.use_direct_estimate = use_observed_error
        self.camera_angles = camera_angles
        self.angle_error = angle_error
        self.padding_fraction = padding_fraction

    @utils.enable_passthrough
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Get the spatial error from the sensor data.

        This currently parameterizes the estimates using the
        geotransforms, which are anchored by x_min and y_max.
        However, the sensor x and y are more-closely related to
        x_center and y_center. Parameterizing by those would make
        more sense, but would require tracking the differences.

        Parameters
        ----------
        X : pd.DataFrame
            Sensor coords and filepaths.
        y : pd.DataFrame
            Geotransforms identifying location.

        Returns
        -------
        self : object
            Returns self.
        """
        utils.check_df_input(
            X,
            self.required_columns,
        )

        # Calculate offsets
        widths = y["pixel_width"] * y["x_size"]
        heights = -y["pixel_height"] * y["y_size"]

        # Estimate values that are just averages
        self.width_ = np.nanmedian(widths)
        self.height_ = np.nanmedian(heights)
        self.pixel_width_ = np.nanmedian(y["pixel_width"])
        self.pixel_height_ = np.nanmedian(y["pixel_height"])
        self.x_rot_ = np.nanmedian(y["x_rot"])
        self.y_rot_ = np.nanmedian(y["y_rot"])
        self.x_size_ = np.round(self.width_ / self.pixel_width_).astype(int)
        self.y_size_ = np.round(np.abs(self.height_ / self.pixel_height_)).astype(int)

        # Estimate spatial error
        X["offset"] = np.sqrt(
            (X["sensor_x"] - y["x_center"]) ** 2.0
            + (X["sensor_y"] - y["y_center"]) ** 2.0
        )
        X_cam = X.groupby("camera_num")
        self.spatial_error_ = X_cam["offset"].mean()

        # `fit` should always return `self`
        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Estimate the search regions.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame with sensor coordinates.

        Returns
        -------
        pd.DataFrame
            DataFrame now with estimated search regions. New columns:
            - x_min, x_max, y_min, y_max, x_center, y_center, spatial_eror, padding
        """

        check_is_fitted(self, "is_fitted_")
        utils.check_df_input(
            X,
            self.required_columns,
        )

        # Calculate properties
        set_manually = [
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "x_center",
            "y_center",
            "spatial_error",
            "padding",
        ]
        for key in preprocessors.GEOTRANSFORM_COLS:
            if key in set_manually:
                continue
            X[key] = getattr(self, key + "_")
        X["x_min"] = X["sensor_x"] - 0.5 * self.width_
        X["x_max"] = X["sensor_x"] + 0.5 * self.width_
        X["y_min"] = X["sensor_y"] - 0.5 * self.height_
        X["y_max"] = X["sensor_y"] + 0.5 * self.height_
        X["x_center"] = X["sensor_x"]
        X["y_center"] = X["sensor_y"]

        # Estimate spatial error
        X["spatial_error"] = np.nan
        # First, we identify what cameras we can use a direct inference for.
        if hasattr(self, "spatial_error_") and self.use_direct_estimate:
            has_direct_estimate = X["camera_num"].isin(self.spatial_error_.index)
            X.loc[has_direct_estimate, "spatial_error"] = self.spatial_error_.loc[
                X.loc[has_direct_estimate, "camera_num"]
            ].values
        # Second, fall-back to expected values based on camera angles
        is_na = X["spatial_error"].isna()
        if is_na.sum() > 0:
            camera_angles = X["camera_num"].map(self.camera_angles)
            X.loc[is_na, "spatial_error"] = X.loc[is_na, "mAltitude"] * np.tan(
                (camera_angles + self.angle_error) * np.pi / 180.0
            )

        # Add padding
        X["padding"] = self.padding_fraction * X["spatial_error"]

        # Ensure correct type
        X[preprocessors.GEOTRANSFORM_COLS] = X[preprocessors.GEOTRANSFORM_COLS].astype(
            float
        )

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Same as predict.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame with sensor coordinates.

        Returns
        -------
        pd.DataFrame
            DataFrame now with estimated search regions. New columns:
            - x_min, x_max, y_min, y_max, x_center, y_center, spatial_eror, padding
        """
        return self.predict(X)

    def score_samples(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
        """Score the samples, according to the error in the predicted center compared
        to the actual center.

        Parameters
        ----------
        X : pd.DataFrame
            The input features used for prediction.

        y : pd.DataFrame
            The dataframe containing the actual center coordinates.

        Returns
        -------
        pd.Series
            The scores representing the error in the predicted center
            compared to the actual center.
        """

        check_is_fitted(self, "is_fitted_")

        y_pred = self.predict(X)
        pred_xs = 0.5 * (y_pred["x_min"] + y_pred["x_max"])
        pred_ys = 0.5 * (y_pred["y_min"] + y_pred["y_max"])

        actual_xs = 0.5 * (y["x_min"] + y["x_max"])
        actual_ys = 0.5 * (y["y_min"] + y["y_max"])

        self.scores_ = np.sqrt(
            (pred_xs - actual_xs) ** 2.0 + (pred_ys - actual_ys) ** 2.0
        )

        return self.scores_

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """Return the median score of the samples.

        Parameters
        ----------
        X : pd.DataFrame
            The input features used for prediction.

        y : pd.DataFrame
            The dataframe containing the actual center coordinates.

        Returns
        -------
        float
            The median score of the samples.
        """

        return np.median(self.score_samples(X, y))
