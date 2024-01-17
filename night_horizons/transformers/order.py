import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SensorAndDistanceOrder(TransformerMixin, BaseEstimator):
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
        self.apply = apply
        self.sensor_order_col = sensor_order_col
        self.sensor_order_map = sensor_order_map
        self.coords_cols = coords_cols

    def fit(self, X, y=None):

        # Center defaults to the first training sample
        self.center_ = X[self.coords_cols].iloc[0]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X['sensor_order'] = X[self.sensor_order_col].map(self.sensor_order_map)

        offset = X[self.coords_cols] - self.center_
        X['d_to_center'] = np.linalg.norm(offset, axis=1)

        # Actual sort
        X_iter = X.sort_values(['sensor_order', 'd_to_center'])
        X_iter['order'] = np.arange(len(X_iter))

        if self.apply:
            return X_iter

        X['order'] = X_iter.loc[X.index, 'order']

        return X