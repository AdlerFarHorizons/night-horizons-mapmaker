import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QueryFilter(TransformerMixin, BaseEstimator):

    def __init__(self, condition):
        self.condition = condition

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):

        X_out = X.query(self.condition)

        return X_out


class Filter(TransformerMixin, BaseEstimator):
    '''Simple estimator to implement easy filtering of rows.
    Does not actually remove rows, but instead adds a `selected` column.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, condition, apply=True):
        self.condition = condition
        self.apply = apply

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        meets_condition = self.condition(X)

        if self.apply:
            return X.loc[meets_condition]

        if 'selected' in X.columns:
            X['selected'] = X['selected'] & meets_condition
        else:
            X['selected'] = meets_condition
        return X


class SteadyFilter(Filter):

    def __init__(
        self,
        column: str = 'imuGyroMag',
        max_gyro: float = 0.075,
    ):

        self.column = column
        self.max_gyro = max_gyro

        def condition(X):
            return X[self.column] < max_gyro

        super().__init__(condition)


class AltitudeFilter(Filter):

    def __init__(
        self,
        column: str = 'mAltitude',
        float_altitude: float = 13000.,
    ):

        self.column = column
        self.float_altitude = float_altitude

        def condition(X):
            return X[column] > float_altitude

        super().__init__(condition)
