import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


from abc import abstractmethod


from sklearn.pipeline import Pipeline


class BaseImageTransformer(TransformerMixin, BaseEstimator):
    '''Transformer for image data.

    Parameters
    ----------
    Returns
    -------
    '''

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):

        X_t = []
        for img in X:
            img_t = self.transform_image(img)
            X_t.append(img_t)

        return X_t

    @abstractmethod
    def transform_image(self, img):
        pass


class PassImageTransformer(BaseImageTransformer):

    def transform_image(self, img):

        return img


class LogscaleImageTransformer(BaseImageTransformer):

    def transform_image(self, img):

        assert np.issubdtype(img.dtype, np.integer), \
            'logscale_img_transform not implemented for imgs with float dtype.'

        # Transform the image
        # We add 1 because log(0) = nan.
        # We have to convert the image first because otherwise max values
        # roll over
        logscale_img = np.log10(img.astype(np.float32) + 1)

        # Scale
        dtype_max = np.iinfo(img.dtype).max
        logscale_img *= dtype_max / np.log10(dtype_max + 1)

        return logscale_img.astype(img.dtype)


class CleanImageTransformer(BaseImageTransformer):

    def __init__(self, fraction=0.03):
        self.fraction = fraction

    def transform_image(self, img):

        img = copy.copy(img)

        assert np.issubdtype(img.dtype, np.integer), \
            'floor not implemented for imgs with float dtype.'

        value = int(self.fraction * np.iinfo(img.dtype).max)
        img[img <= value] = 0

        return img


CLEAN_LOGSCALE_IMAGE_PIPELINE = Pipeline([
    ('clean', CleanImageTransformer()),
    ('logscale', LogscaleImageTransformer()),
])