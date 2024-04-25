from abc import abstractmethod
import copy

import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import pyproj
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..data_io import GDALDatasetIO
from ..exceptions import OutOfBoundsError


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


class RasterCoordinateTransformer(TransformerMixin, BaseEstimator):
    '''Transforms physical coordinates to/from pixel coordinates.
    Assumes Cartesian coordinates! Distances don't work otherwise.

    Parameters
    ----------
    Returns
    -------
    '''

    def fit(
        self,
        X,
        y=None,
        pixel_width: float = None,
        pixel_height: float = None,
        padding: float = None,
        crs: pyproj.CRS = None,
    ):

        if padding is None:
            padding = X['padding'].max()

        # Get bounds
        self.x_min_ = X['x_min'].min() - padding
        self.x_max_ = X['x_max'].max() + padding
        self.y_min_ = X['y_min'].min() - padding
        self.y_max_ = X['y_max'].max() + padding

        # Pixel resolution
        if pixel_width is None:
            self.pixel_width_ = np.median(X['pixel_width'])
        else:
            self.pixel_width_ = pixel_width
        if pixel_height is None:
            self.pixel_height_ = np.median(X['pixel_height'])
        else:
            self.pixel_height_ = pixel_height

        # Get dimensions
        width = self.x_max_ - self.x_min_
        self.x_size_ = int(np.round(width / self.pixel_width_))
        height = self.y_max_ - self.y_min_
        self.y_size_ = int(np.round(height / -self.pixel_height_))

        # Re-record pixel values to account for rounding
        self.pixel_width_ = width / self.x_size_
        self.pixel_height_ = -height / self.y_size_

        self.crs_ = crs

        return self

    def fit_to_dataset(self, dataset: gdal.Dataset):

        if dataset is None:
            raise TypeError('dataset must be provided.')

        (
            (self.x_min_, self.x_max_),
            (self.y_min_, self.y_max_),
            self.pixel_width_,
            self.pixel_height_,
        ) = GDALDatasetIO.get_bounds_from_dataset(dataset)
        self.x_size_ = dataset.RasterXSize
        self.y_size_ = dataset.RasterYSize
        self.crs_ = pyproj.CRS(dataset.GetProjection())

        return self

    def transform(self, X, direction='to_pixel'):

        if direction == 'to_pixel':
            X_t = self.transform_to_pixel(X)
        elif direction == 'to_physical':
            X_t = self.transform_to_physical(X)
        else:
            raise ValueError(
                f'direction must be "to_pixel" or "to_physical", '
                f'not "{direction}"'
            )

        return X_t

    def physical_to_pixel(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
    ):
        '''
        Parameters
        ----------
        Returns
        -------
        '''

        # Get physical dimensions
        x_imgframe = x_min - self.x_min_
        y_imgframe = self.y_max_ - y_max
        width = x_max - x_min
        height = y_max - y_min

        # Convert to pixels
        x_off = np.round(x_imgframe / self.pixel_width_)
        y_off = np.round(y_imgframe / -self.pixel_height_)
        x_size = np.round(width / self.pixel_width_)
        y_size = np.round(height / -self.pixel_height_)

        try:
            # Change dtypes
            x_off = x_off.astype(int)
            y_off = y_off.astype(int)
            x_size = x_size.astype(int)
            y_size = y_size.astype(int)

        # When we're passing in single values.
        except TypeError:
            # Change dtypes
            x_off = int(x_off)
            y_off = int(y_off)
            x_size = int(x_size)
            y_size = int(y_size)

        return x_off, y_off, x_size, y_size

    def pixel_to_physical(self, x_off, y_off, x_size, y_size):

        # Convert to physical units.
        x_imgframe = x_off * self.pixel_width_
        y_imgframe = y_off * -self.pixel_height_
        width = x_size * self.pixel_width_
        height = y_size * -self.pixel_height_

        # Convert to bounds
        x_min = x_imgframe + self.x_min_
        y_max = self.y_max_ - y_imgframe
        x_max = x_min + width
        y_min = y_max - height

        return x_min, x_max, y_min, y_max

    def check_bounds(self, x_off, y_off, x_size, y_size):

        # Validate
        oob = (
            (x_off < 0)
            | (y_off < 0)
            | (x_off + x_size > self.x_size_)
            | (y_off + y_size > self.y_size_)
        )
        if isinstance(oob, bool):
            if oob:
                raise OutOfBoundsError(
                    'Tried to convert physical to pixels, but '
                    'the provided coordinates are outside the bounds '
                    'of the raster dataset'
                )
        else:
            n_oob = oob.sum()
            if n_oob > 0:
                raise OutOfBoundsError(
                    'Tried to convert physical to pixels, but '
                    f'{n_oob} of {oob.size} are outside the bounds '
                    'of the raster dataset'
                )

    def handle_out_of_bounds(self, x_off, y_off, x_size, y_size, trim=False):

        # By default we raise an error
        if not trim:
            self.check_bounds(x_off, y_off, x_size, y_size)

        # But we can also trim
        else:

            x_off = copy.copy(x_off)
            y_off = copy.copy(y_off)
            x_size = copy.copy(x_size)
            y_size = copy.copy(y_size)

            try:
                # Handle out-of-bounds
                x_off[x_off < 0] = 0
                y_off[y_off < 0] = 0
                x_size[x_off + x_size > self.x_size_] = self.x_size_ - x_off
                y_size[y_off + y_size > self.y_size_] = self.y_size_ - y_off

            except TypeError:
                # Handle out-of-bounds
                if x_off < 0:
                    x_off = 0
                elif x_off + x_size > self.x_size_:
                    x_size = self.x_size_ - x_off
                if y_off < 0:
                    y_off = 0
                elif y_off + y_size > self.y_size_:
                    y_size = self.y_size_ - y_off

        return x_off, y_off, x_size, y_size

    def transform_to_pixel(self, X, trim: bool = False):

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
        )

        # Check nothing is oob
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.handle_out_of_bounds(
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size'],
            trim=trim,
        )

        return X

    def transform_to_physical(self, X):

        (
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
        ) = self.pixel_to_physical(
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        )

        return X
