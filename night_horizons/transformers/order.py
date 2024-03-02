import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OrderTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, order_columns, apply=True, ascending=True):

        self.order_columns = order_columns
        self.apply = apply
        self.ascending = ascending

    def fit(self, X, y=None):

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):

        # Actual sort
        X_sorted = X.sort_values(self.order_columns, ascending=self.ascending)
        X_sorted['order'] = np.arange(len(X_sorted))

        if self.apply:
            return X_sorted

        X['order'] = X_sorted.loc[X.index, 'order']

        return X


class SensorAndDistanceOrder(OrderTransformer):
    '''Simple estimator to implement ordering of data.
    For consistency with other transformers, does not actually rearrange data.
    Instead, adds a column `order` that indicates the order to take.

    The center defaults to that of the first training sample.

    TODO: Breaking this up into multiple individual transforms makes sense,
        if this is something the user is expected to experiment with.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        apply=True,
        sensor_order_col='camera_num',
        sensor_order_map={0: 1, 1: 0, 2: 2},
        coords_cols=['x_center', 'y_center'],
    ):
        self.sensor_order_col = sensor_order_col
        self.sensor_order_map = sensor_order_map
        self.coords_cols = coords_cols

        super().__init__(
            apply=apply,
            order_columns=['sensor_order', 'd_to_center']
        )

    def fit(self, X, y=None):

        # Center defaults to the first training sample
        self.center_ = X[self.coords_cols].iloc[0]
        self.is_fitted_ = True
        return self

    def transform(self, X):

        X['sensor_order'] = X[self.sensor_order_col].map(self.sensor_order_map)

        offset = X[self.coords_cols] - self.center_
        X['d_to_center'] = np.linalg.norm(offset, axis=1)

        return super().transform(X)
