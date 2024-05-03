"""Module for transforming the data by ordering it.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OrderTransformer(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        order_columns: list,
        apply: bool = True,
        ascending: bool = True,
    ):
        """
        Initialize the OrderTransformer class.

        Parameters
        ----------
        order_columns : list
            A list of column names to order the data by,
            passed to pd.DataFrame.sort_values.
        apply : bool, optional
            Flag indicating whether to apply the ordering or not. Default is True.
            If False, just add the "order" column.
        ascending : bool, optional
            Flag indicating whether to sort the data in ascending order or not.
            Default is True.
        """
        self.order_columns = order_columns
        self.apply = apply
        self.ascending = ascending

    def fit(self, X: pd.DataFrame, y=None) -> "OrderTransformer":
        """Fitting is a no-op, i.e. nothing is done.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer on.

        y : None, optional
            The target variable. This parameter is ignored in this method.

        Returns
        -------
        OrderTransformer
            The fitted transformer instance.
        """

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by sorting it based on the specified order columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with the specified order columns sorted,
            or just with an added "order" column if self.apply is False.
        """

        # Actual sort
        X_sorted = X.sort_values(self.order_columns, ascending=self.ascending)
        X_sorted["order"] = np.arange(len(X_sorted))

        if self.apply:
            return X_sorted

        X["order"] = X_sorted.loc[X.index, "order"]

        return X


class SensorAndDistanceOrder(OrderTransformer):
    """Simple estimator to implement ordering of data according to sensor and distance.

    The center defaults to that of the first training sample.
    """

    def __init__(
        self,
        apply: bool = True,
        sensor_order_col: str = "camera_num",
        sensor_order_map: dict = {0: 1, 1: 0, 2: 2},
        coords_cols: list[str] = ["x_center", "y_center"],
    ):
        """
        Initialize the OrderTransformer class.

        Parameters
        ----------
        apply : bool, optional
            Flag indicating whether to apply the transformer, by default True.
        sensor_order_col : str, optional
            Name of the column containing sensor order information,
            by default "camera_num".
        sensor_order_map : dict, optional
            Mapping of sensor order values to new order values,
            by default {0: 1, 1: 0, 2: 2}.
        coords_cols : list[str], optional
            List of column names containing coordinates information,
            by default ["x_center", "y_center"].
        """
        self.sensor_order_col = sensor_order_col
        self.sensor_order_map = sensor_order_map
        self.coords_cols = coords_cols

        super().__init__(apply=apply, order_columns=["sensor_order", "d_to_center"])

    def fit(self, X: pd.DataFrame, y=None) -> "SensorAndDistanceOrder":
        """Fits the transformer to the input data, i.e. sets the center.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer on.

        y : None, optional
            The target variable. This parameter is ignored.

        Returns
        -------
        SensorAndDistanceOrder
            The fitted transformer object.
        """
        # Center defaults to the first training sample
        self.center_ = X[self.coords_cols].iloc[0]
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by adding additional columns, and then
        ordering by them.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """

        X["sensor_order"] = X[self.sensor_order_col].map(self.sensor_order_map)

        offset = X[self.coords_cols] - self.center_
        X["d_to_center"] = np.linalg.norm(offset, axis=1)

        return super().transform(X)
