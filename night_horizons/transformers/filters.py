"""Module for transforming the data by filtering it (cutting out some data)."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QueryFilter(TransformerMixin, BaseEstimator):
    """Class for filtering data based on a call to pd.DataFrame.query."""

    def __init__(self, condition: str):
        """
        Initializes a new instance of the QueryFilter class.

        Parameters
        ----------
        condition : str
            The condition to be applied by the filter.
        """
        self.condition = condition

    def fit(self, X: pd.DataFrame, y=None) -> "QueryFilter":
        """Fitting is a no-op, i.e. nothing is done.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer on.

        y : None, optional
            The target variable. This parameter is ignored in this method.

        Returns
        -------
        QueryFilter
            The fitted transformer instance.
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the query.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to apply the query on.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame after applying the query.
        """

        X_out = X.query(self.condition)

        return X_out


class Filter(TransformerMixin, BaseEstimator):
    """Simple estimator to implement easy filtering of rows.
    """

    def __init__(self, condition: callable, apply: bool = True):
        """
        Initializes a Filter object.

        Parameters
        ----------
        condition : callable
            The condition function that determines what is filtered.
            should be applied or not.
        apply : bool, optional
            Flag indicating whether the filter should be applied. If set
            to False, the filter will only add a `selected` column to the
            input DataFrame. Default is True.
        """
        self.condition = condition
        self.apply = apply

    def fit(self, X: pd.DataFrame, y=None) -> "Filter":
        """Fitting is a no-op, i.e. nothing is done.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer on.

        y : None, optional
            The target variable. This parameter is ignored in this method.

        Returns
        -------
        QueryFilter
            The fitted transformer instance.
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the filter to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to apply the filter on.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame based on the applied filter.
        """
        meets_condition = self.condition(X)

        if self.apply:
            return X.loc[meets_condition]

        if "selected" in X.columns:
            X["selected"] = X["selected"] & meets_condition
        else:
            X["selected"] = meets_condition
        return X


class SteadyFilter(Filter):
    """Class for filtering out images that are moving too quickly."""

    def __init__(
        self,
        column: str = "imuGyroMag",
        max_gyro: float = 0.075,
    ):
        """
        Initialize a SteadyFilter object.

        Parameters
        ----------
        column : str, optional
            The column name to filter on. Default is 'imuGyroMag'.
        max_gyro : float, optional
            The maximum gyro value allowed. Default is 0.075.

        Returns
        -------
        None
        """

        self.column = column
        self.max_gyro = max_gyro

        def condition(X):
            return X[self.column] < max_gyro

        super().__init__(condition)


class AltitudeFilter(Filter):
    """Class for filtering out images that are not at the float altitude."""

    def __init__(
        self,
        column: str = "mAltitude",
        float_altitude: float = 13000.0,
    ):
        """
        Initialize an AltitudeFilter object.

        Parameters
        ----------
        column : str, optional
            The column name to filter on. Default is 'mAltitude'.
        float_altitude : float, optional
            The minimum altitude allowed. Default is 13000
        """

        self.column = column
        self.float_altitude = float_altitude

        def condition(X):
            return X[column] > float_altitude

        super().__init__(condition)
