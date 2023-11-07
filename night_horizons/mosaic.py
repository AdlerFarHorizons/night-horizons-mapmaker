import glob
import os
from typing import Union
import warnings

import cv2
import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from . import preprocess, utils


class ReferencedMosaic(TransformerMixin, BaseEstimator):
    '''Assemble a mosaic from georeferenced images.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        passthrough: bool = False,
    ):
        self.filepath = filepath
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fill_value = fill_value
        self.dtype = dtype
        self.n_bands = n_bands
        self.passthrough = passthrough

    # DEBUG
    # def open(self, filename: str, crs: pyproj.CRS = None, *args, **kwargs):

    #     dataset.dataset = gdal.Open(filename, *args, **kwargs)

    #     # CRS handling
    #     if isinstance(crs, str):
    #         crs = pyproj.CRS(crs)
    #     if crs is None:
    #         crs = pyproj.CRS(dataset.dataset.GetProjection())
    #     else:
    #         dataset.dataset.SetProjection(crs.to_wkt())
    #     dataset.crs = crs
    #     dataset.filename = filename

    #     # Get bounds
    #     (
    #         dataset.x_bounds,
    #         dataset.y_bounds,
    #         dataset.pixel_width,
    #         dataset.pixel_height
    #     ) = get_bounds_from_dataset(
    #         dataset.dataset,
    #         crs,
    #     )

    #     return dataset

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        '''The main thing the fitting does is create an empty dataset to hold
        the mosaic.

        Parameters
        ----------
        X
            A dataframe containing the bounds of each added image.

        y
            Empty.

        Returns
        -------
        self
            Returns self.
        '''

        # Check the input is good.
        X = utils.check_df_input(
            X,
            ['filepath'] + preprocess.GEOBOUNDS_COLS,
            passthrough=self.passthrough
        )
        if os.path.isfile(self.filepath):
            raise FileExistsError('File already exists at destination.')

        # Convert CRS as needed
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Get bounds
        self.x_min_ = X['x_min'].min()
        self.x_max_ = X['x_max'].max()
        self.y_min_ = X['y_min'].min()
        self.y_max_ = X['y_max'].max()

        # Pixel resolution
        if self.pixel_width is None:
            self.pixel_width_ = np.median(X['pixel_width'])
        else:
            self.pixel_width_ = self.pixel_width
        if self.pixel_height is None:
            self.pixel_height_ = np.median(X['pixel_height'])
        else:
            self.pixel_height_ = self.pixel_height

        # Get dimensions
        width = self.x_max_ - self.x_min_
        xsize = int(np.round(width / self.pixel_width_))
        height = self.y_max_ - self.y_min_
        ysize = int(np.round(height / -self.pixel_height_))

        # Re-record pixel values to account for rounding
        self.pixel_width_ = width / xsize
        self.pixel_height_ = height / ysize

        # Initialize an empty GeoTiff
        driver = gdal.GetDriverByName('GTiff')
        self.dataset_ = driver.Create(
            self.filepath,
            xsize=xsize,
            ysize=ysize,
            bands=self.n_bands,
            options=['TILED=YES']
        )

        # Properties
        self.dataset_.SetProjection(self.crs.to_wkt())
        self.dataset_.SetGeoTransform([
            self.x_min_,
            self.pixel_width_,
            0.,
            self.y_max_,
            0.,
            self.pixel_height_,
        ])
        if self.n_bands == 4:
            self.dataset_.GetRasterBand(4).SetMetadataItem('Alpha', '1')

        return self

    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        X = utils.check_df_input(
            X,
            ['filepath'] + preprocess.GEOBOUNDS_COLS,
            passthrough=self.passthrough
        )

        # Check if fit had been called
        check_is_fitted(self, 'dataset_')

        # Loop through and include
        for i, fp in enumerate(tqdm.tqdm(X['filepath'])):

            row = X.loc[i]

            # Get data
            src_img = utils.load_image(
                row['filepath'],
                dtype=self.dtype,
            )
            dst_img = self.get_img(
                row['x_min'],
                row['x_max'],
                row['y_min'],
                row['y_max'],
            )

            # Resize the source image
            src_img_resized = cv2.resize(
                src_img,
                (dst_img.shape[1], dst_img.shape[0])
            )

            # Combine the images
            blended_img = self.blend_images(
                src_img=src_img_resized,
                dst_img=dst_img,
                fill_value=self.fill_value,
            )

            # Store the image
            self.save_img(
                blended_img,
                row['x_min'],
                row['x_max'],
                row['y_min'],
                row['y_max'],
            )

        # Finish by flushing the cache
        self.dataset_.FlushCache()

    def bounds_to_offset(self, x_min, x_max, y_min, y_max):

        # Get offsets
        x_offset = x_min - self.x_min_
        x_offset_count = int(np.round(x_offset / self.pixel_width_))
        y_offset = y_min - self.y_min_
        y_offset_count = int(np.round(y_offset / self.pixel_height_))

        # Get width counts
        xsize = int(np.round((x_max - x_min) / self.pixel_width_))
        ysize = int(np.round((y_max - y_min) / self.pixel_height_))

        return x_offset_count, y_offset_count, xsize, ysize

    def get_img(self, x_min, x_max, y_min, y_max):

        # Out of bounds
        if (
            (x_min > self.x_max_)
            or (x_max < self.x_min_)
            or (y_min > self.y_max_)
            or (y_max < self.y_min_)
        ):
            raise ValueError(
                'Tried to retrieve data fully out-of-bounds.'
            )

        # Only partially out-of-bounds
        if x_min < self.x_min_:
            x_min = self.x_min_
        if x_max > self.x_max_:
            x_max = self.x_max_
        if y_min < self.y_min_:
            y_min = self.y_min_
        if y_max > self.y_max_:
            y_max = self.y_max_

        x_offset_count, y_offset_count, xsize, ysize = self.bounds_to_offset(
            x_min, x_max, y_min, y_max
        )

        img = self.dataset_.ReadAsArray(
            xoff=x_offset_count,
            yoff=y_offset_count,
            xsize=xsize,
            ysize=ysize
        )
        return img.transpose(1, 2, 0)

    def save_img(self, img, x_min, x_max, y_min, y_max):

        x_offset_count, y_offset_count, xsize, ysize = self.bounds_to_offset(
            x_min, x_max, y_min, y_max
        )

        img_to_save = img.transpose(2, 0, 1)
        self.dataset_.WriteArray(
            img_to_save,
            xoff=x_offset_count,
            yoff=y_offset_count
        )

    @staticmethod
    def blend_images(src_img, dst_img, fill_value=None):

        # Fill value defaults to values that would be opaque
        if fill_value is None:
            if np.issubdtype(dst_img.dtype, np.integer):
                fill_value = 255
            else:
                fill_value = 1.

        # Blend
        # Doesn't consider zeros in the final channel as empty
        n_bands = dst_img.shape[-1]
        is_empty = (dst_img[:, :, :n_bands - 1].sum(axis=2) == 0)
        blended_img = []
        for j in range(n_bands):
            try:
                blended_img_j = np.where(
                    is_empty,
                    src_img[:, :, j],
                    dst_img[:, :, j]
                )
            # When there's no band information in the one we're blending,
            # fall back to the fill value
            except IndexError:
                blended_img_j = np.full(
                    dst_img.shape[:2],
                    fill_value,
                    dtype=dst_img.dtype
                )
            blended_img.append(blended_img_j)
        blended_img = np.array(blended_img).transpose(1, 2, 0)

        return blended_img
