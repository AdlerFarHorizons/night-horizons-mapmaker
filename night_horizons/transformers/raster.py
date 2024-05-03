"""Module for transforming raster data."""

from abc import abstractmethod
import copy
from typing import Tuple, Union

import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..data_io import GDALDatasetIO
from ..exceptions import OutOfBoundsError

gdal.UseExceptions()


class BaseImageTransformer(TransformerMixin, BaseEstimator):
    """Base class transformer for image data."""

    def fit(self, X: list[np.ndarray], y=None) -> "BaseImageTransformer":
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

    def transform(self, X: list[np.ndarray]) -> list[np.ndarray]:
        """Perform the transformation defined in "transform_image" on each image in X.

        Parameters
        ----------
        X : list[np.ndarray]
            A list of numpy arrays representing the input images.

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays representing the transformed images.
        """

        X_t = []
        for img in X:
            img_t = self.transform_image(img)
            X_t.append(img_t)

        return X_t

    @abstractmethod
    def transform_image(self, img: np.ndarray) -> np.ndarray:
        """Abstract method for applying a transform, overwritten by subclasses.

        Parameters
        ----------
        img : np.ndarray
            The input image to be transformed.

        Returns
        -------
        np.ndarray
            The transformed image.
        """


class PassImageTransformer(BaseImageTransformer):
    """Class for not altering images. This exists for compatibility with the pipeline."""

    def transform_image(self, img: np.ndarray) -> np.ndarray:
        """Return the image unchanged.

        Parameters
        ----------
        img : np.ndarray
            The input image to be transformed.

        Returns
        -------
        np.ndarray
            The transformed image.
        """

        return img


class LogscaleImageTransformer(BaseImageTransformer):
    """Class for logscaling images."""

    def transform_image(self, img: np.ndarray) -> np.ndarray:
        """Logscale the image. We add 1 to the image before taking the log
        to avoid log(0).

        Parameters
        ----------
        img : np.ndarray
            The input image to be transformed.

        Returns
        -------
        np.ndarray
            The transformed image with log scaling applied.
        """

        assert np.issubdtype(
            img.dtype, np.integer
        ), "logscale_img_transform not implemented for imgs with float dtype."

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
    """Class for cleaning images by setting all values below a threshold to 0."""

    def __init__(self, fraction: float = 0.03):
        """
        Initialize the CleanImageTransformer object.

        Parameters
        ----------
        fraction : float, optional
            Fraction of the maximum pixel value to set as the threshold,
            by default 0.03. Below this threshold, pixel values are set to 0.

        Returns
        -------
        None
        """
        self.fraction = fraction

    def transform_image(self, img: np.ndarray) -> np.ndarray:
        """Transform the image by setting all values below a threshold to 0.

        Parameters
        ----------
        img : np.ndarray
            The input image to be transformed.

        Returns
        -------
        np.ndarray
            The transformed image with values below the threshold set to 0.
        """

        img = copy.copy(img)

        assert np.issubdtype(
            img.dtype, np.integer
        ), "floor not implemented for imgs with float dtype."

        value = int(self.fraction * np.iinfo(img.dtype).max)
        img[img <= value] = 0

        return img


CLEAN_LOGSCALE_IMAGE_PIPELINE = Pipeline(
    [
        ("clean", CleanImageTransformer()),
        ("logscale", LogscaleImageTransformer()),
    ]
)
"""Image transformer that consists of a CleanImageTransformer followed by a
LogscaleImageTransformer.
"""


class RasterCoordinateTransformer(TransformerMixin, BaseEstimator):
    """Class for transforming a rectangle of physical coordinates
    to/from pixel coordinates. Assumes Cartesian coordinates!
    Distances don't work otherwise.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        pixel_width: float = None,
        pixel_height: float = None,
        padding: float = None,
        crs: pyproj.CRS = None,
    ) -> "RasterCoordinateTransformer":
        """
        Fits the RasterCoordinateTransformer to the given data. This sets all
        the parameters needed for converting between physical (e.g. x_min)
        and pixel.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing the coordinates and other relevant information.
            Required columns:
            - x_min: The minimum x-coordinate.
            - x_max: The maximum x-coordinate.
            - y_min: The minimum y-coordinate.
            - y_max: The maximum y-coordinate.
            - pixel_width: The width of a pixel.
            - pixel_height: The height of a pixel.
            - padding: The padding value to be added to the coordinates.
        y : None, optional
            The target variable (not used in this method), by default None.
        pixel_width : float, optional
            The desired pixel width, by default None.
        pixel_height : float, optional
            The desired pixel height, by default None.
        padding : float, optional
            The padding value to be added to the coordinates, by default None.
        crs : pyproj.CRS, optional
            The coordinate reference system, by default None.

        Returns
        -------
        RasterCoordinateTransformer
            The fitted RasterCoordinateTransformer object.
        """

        if padding is None:
            padding = X["padding"].max()

        # Get bounds
        self.x_min_ = X["x_min"].min() - padding
        self.x_max_ = X["x_max"].max() + padding
        self.y_min_ = X["y_min"].min() - padding
        self.y_max_ = X["y_max"].max() + padding

        # Pixel resolution
        if pixel_width is None:
            self.pixel_width_ = np.median(X["pixel_width"])
        else:
            self.pixel_width_ = pixel_width
        if pixel_height is None:
            self.pixel_height_ = np.median(X["pixel_height"])
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

    def fit_to_dataset(self, dataset: gdal.Dataset) -> "RasterCoordinateTransformer":
        """Alternative to fit that takes a GDAL dataset as input.

        This method fits the raster coordinate transformer to the provided GDAL dataset.
        It calculates the bounds, pixel size, size of the dataset, and CRS
        based on the information from the dataset.

        Parameters
        ----------
        dataset : gdal.Dataset
            The GDAL dataset to fit the transformer to.

        Returns
        -------
        RasterCoordinateTransformer
            The fitted RasterCoordinateTransformer object.

        Raises
        ------
        TypeError
            If the dataset is None.
        """

        if dataset is None:
            raise TypeError("dataset must be provided.")

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

    def transform(self, X: pd.DataFrame, direction: str = "to_pixel") -> pd.DataFrame:
        """Transform a dataframe to/from pixel coordinates, i.e. adds new columns
        for the pixel/physical coordinates. The dataframe must contain appropriate
        columns corresponding to rectangular regions.

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe to be transformed.

        direction : str, optional
            The direction of the transformation. Default is 'to_pixel'.
            Valid values are 'to_pixel' and 'to_physical'.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe with added columns for pixel/physical coordinates.
        """

        if direction == "to_pixel":
            X_t = self.transform_to_pixel(X)
        elif direction == "to_physical":
            X_t = self.transform_to_physical(X)
        else:
            raise ValueError(
                f'direction must be "to_pixel" or "to_physical", ' f'not "{direction}"'
            )

        return X_t

    def physical_to_pixel(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> Tuple[int, int, int, int]:
        """Convert the physical bounds of a rectangle to pixel coordinates.

        Parameters
        ----------
        x_min : float
            The minimum x-coordinate of the physical bounds.
        x_max : float
            The maximum x-coordinate of the physical bounds.
        y_min : float
            The minimum y-coordinate of the physical bounds.
        y_max : float
            The maximum y-coordinate of the physical bounds.

        Returns
        -------
        Tuple[int, int, int, int]
            A tuple containing the x-offset, y-offset, width,
            and height in pixel coordinates.
        """

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

    def pixel_to_physical(
        self,
        x_off: int,
        y_off: int,
        x_size: int,
        y_size: int,
    ) -> Tuple[float, float, float, float]:
        """
        Converts a the pixel offsets and sizes of a rectangle to physical bounds.

        Parameters
        ----------
        x_off : int
            The x-offset of the pixel.
        y_off : int
            The y-offset of the pixel.
        x_size : int
            The number of pixels in the x-direction.
        y_size : int
            The number of pixels in the y-direction.

        Returns
        -------
        Tuple[float, float, float, float]
            A tuple containing the minimum and maximum x and y coordinates
            in physical units.
        """

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

    def transform_to_pixel(self, X: pd.DataFrame, trim: bool = False) -> pd.DataFrame:
        """Transform a series of rectangles in physical space to pixel space.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the coordinates to be transformed.
            Must include the following columns:
            - x_min: The minimum x-coordinate.
            - x_max: The maximum x-coordinate.
            - y_min: The minimum y-coordinate.
            - y_max: The maximum y-coordinate.
        trim : bool, optional
            Specifies whether to trim the coordinates that are out of bounds,
            by default False.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with the coordinates converted to pixel values.
        """

        # Convert to pixels
        (X["x_off"], X["y_off"], X["x_size"], X["y_size"]) = self.physical_to_pixel(
            X["x_min"],
            X["x_max"],
            X["y_min"],
            X["y_max"],
        )

        # Check nothing is oob
        (X["x_off"], X["y_off"], X["x_size"], X["y_size"]) = self.handle_out_of_bounds(
            X["x_off"],
            X["y_off"],
            X["x_size"],
            X["y_size"],
            trim=trim,
        )

        return X

    def transform_to_physical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a series of rectangles in pixel space to physical space.

        This method takes a DataFrame `X` containing information about rectangles in pixel space,
        and transforms their coordinates to physical space using the `pixel_to_physical` method.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing information about rectangles in pixel space.
            It should have the following columns:
            - x_off: x-coordinate of the top-left corner of the rectangle
            - y_off: y-coordinate of the top-left corner of the rectangle
            - x_size: width of the rectangle
            - y_size: height of the rectangle

        Returns
        -------
        pd.DataFrame
            DataFrame with the transformed coordinates in physical space.
            It will have the following additional columns:
            - x_min: minimum x-coordinate in physical space
            - x_max: maximum x-coordinate in physical space
            - y_min: minimum y-coordinate in physical space
            - y_max: maximum y-coordinate in physical space
        """

        (
            X["x_min"],
            X["x_max"],
            X["y_min"],
            X["y_max"],
        ) = self.pixel_to_physical(X["x_off"], X["y_off"], X["x_size"], X["y_size"])

        return X

    def check_bounds(
        self,
        x_off: Union[int, np.ndarray],
        y_off: Union[int, np.ndarray],
        x_size: Union[int, np.ndarray],
        y_size: Union[int, np.ndarray],
    ):
        """Check if pixel coordinates are out of bounds.

        Parameters
        ----------
        x_off : int
            The x-coordinate offset of the pixel.
        y_off : int
            The y-coordinate offset of the pixel.
        x_size : int
            The width of the pixel.
        y_size : int
            The height of the pixel.

        Raises
        ------
        OutOfBoundsError
            If the provided coordinates are outside the bounds of the raster dataset.

        """

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
                    "Tried to convert physical to pixels, but "
                    "the provided coordinates are outside the bounds "
                    "of the raster dataset"
                )
        else:
            n_oob = oob.sum()
            if n_oob > 0:
                raise OutOfBoundsError(
                    "Tried to convert physical to pixels, but "
                    f"{n_oob} of {oob.size} are outside the bounds "
                    "of the raster dataset"
                )

    def handle_out_of_bounds(
        self,
        x_off: Union[int, np.ndarray],
        y_off: Union[int, np.ndarray],
        x_size: Union[int, np.ndarray],
        y_size: Union[int, np.ndarray],
        trim: bool = False,
    ) -> Tuple[
        Union[int, np.ndarray],
        Union[int, np.ndarray],
        Union[int, np.ndarray],
        Union[int, np.ndarray],
    ]:
        """Checks what pixels are in bounds, and trims those that aren't
        if trim is True. Otherwise raises an error.

        Parameters
        ----------
        x_off : int or np.ndarray
            The x-coordinate offset of the pixel.
        y_off : int or np.ndarray
            The y-coordinate offset of the pixel.
        x_size : int or np.ndarray
            The width of the pixel.
        y_size : int or np.ndarray
            The height of the pixel.
        trim : bool, optional
            If True, trims the out-of-bounds pixels. If False, and there are
            out-of-bounds pixels, raises an error.
            Default is False.

        Returns
        -------
        Tuple[
            int or np.ndarray,
            int or np.ndarray,
            int or np.ndarray,
            int or np.ndarray
        ] :
            The updated x_off, y_off, x_size, and y_size values after handling
            out-of-bounds pixels.
        """

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
