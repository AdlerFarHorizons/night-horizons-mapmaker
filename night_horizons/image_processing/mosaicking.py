import copy
import inspect
import os
import pickle
import re
import shutil
import time
import tracemalloc
from typing import Tuple, Union
import warnings

import cv2
import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import scipy
from sklearn.utils.validation import check_is_fitted
import tqdm
import yaml

from night_horizons.exceptions import OutOfBoundsError
from . import operators

from .batch import BatchProcessor

from .. import (
    preprocessors, utils, raster
)
from . import processors


class Mosaicker(BatchProcessor):
    '''Assemble a mosaic from georeferenced images.

    TODO: filepath is a data-dependent parameter, so it really should be
    called at the time of the fit.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        io_manager,
        processor,
        scorer: processors.Processor = None,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        dtype: type = np.uint8,
        fill_value: Union[int, float] = None,
        n_bands: int = 4,
        outline: int = 0,
        log_keys: list[str] = ['ind', 'return_code'],
        passthrough: Union[list[str], bool] = True,
    ):

        super().__init__(
            processor=processor,
            passthrough=passthrough,
            log_keys=log_keys,
            scorer=scorer,
        )

        # Store settings for latter use
        self.io_manager = io_manager
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.dtype = dtype
        self.fill_value = fill_value
        self.n_bands = n_bands
        self.outline = outline

        self.required_columns = ['filepath'] + preprocessors.GEOTRANSFORM_COLS

    @utils.enable_passthrough
    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        i_start: Union[str, int] = 'checkpoint',
        dataset: gdal.Dataset = None,
    ):

        # The fitting that's done for all image processing pipelines
        super().fit(X, y, i_start=i_start)

        # TODO: Make this compatible with dependency injection
        if not isinstance(self.crs, pyproj.CRS):
            self.crs = pyproj.CRS(self.crs)

        # If the dataset was not passed in, load it if possible
        if (
            (
                (self.io_manager.file_exists == 'load')
                and os.path.isfile(self.io_manager.output_filepaths['mosaic'])
            )
            or (self.i_start_ != 0)
        ):
            if dataset is not None:
                raise ValueError(
                    'Cannot both pass in a dataset and load a file')
            dataset = self.io_manager.open_dataset()

        # If we have a loaded dataset by this point, get fit params from it
        if dataset is not None:
            self.get_fit_from_dataset(dataset)

        # Otherwise, make a new dataset
        else:
            if self.i_start_ != 0:
                raise ValueError(
                    'Creating a new dataset, '
                    'but the starting iteration is not 0. '
                    'If creating a new dataset, should start with i = 0.'
                )
            self.create_containing_dataset(X)

        # Fit the processor and scorer too
        # While this is generic and expected for the majority of
        # BatchProcessors, it cannot be part of self.fit because it has to be
        # called after the rest of the fitting is done.
        self.processor.fit(self)
        if self.scorer is not None:
            self.scorer.fit(self)

        return self

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        '''Preprocessing required before doing the full loop.
        This should focus on preprocessing that depends on the particular
        image processor (e.g. for mosaickers this includes putting the
        coordinates in the pixel-based frame of the mosaic).

        Parameters
        ----------
        Returns
        -------
        '''

        X_t = self.transform_to_pixel(X)

        # Get the dataset
        resources = {
            'dataset': self.io_manager.open_dataset(),
        }

        return X_t, resources

    def postprocess(self, X_t, resources):

        # Close out the dataset
        resources['dataset'].FlushCache()
        resources['dataset'] = None

        return X_t

    def get_fit_from_dataset(self, dataset):

        # Get the dataset bounds
        (
            (self.x_min_, self.x_max_),
            (self.y_min_, self.y_max_),
            self.pixel_width_, self.pixel_height_
        ) = raster.get_bounds_from_dataset(
            dataset,
            self.crs,
        )
        self.x_size_ = dataset.RasterXSize
        self.y_size_ = dataset.RasterYSize

        # Close out the dataset for now. (Reduces likelihood of mem leaks.)
        dataset.FlushCache()
        dataset = None

    def create_containing_dataset(self, X):

        # Get bounds
        max_padding = X['padding'].max()
        self.x_min_ = X['x_min'].min() - max_padding
        self.x_max_ = X['x_max'].max() + max_padding
        self.y_min_ = X['y_min'].min() - max_padding
        self.y_max_ = X['y_max'].max() + max_padding

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
        self.x_size_ = int(np.round(width / self.pixel_width_))
        height = self.y_max_ - self.y_min_
        self.y_size_ = int(np.round(height / -self.pixel_height_))

        # Re-record pixel values to account for rounding
        self.pixel_width_ = width / self.x_size_
        self.pixel_height_ = -height / self.y_size_

        # Initialize an empty GeoTiff
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            self.io_manager.output_filepaths['mosaic'],
            xsize=self.x_size_,
            ysize=self.y_size_,
            bands=self.n_bands,
            options=['TILED=YES']
        )

        # Properties
        dataset.SetProjection(self.crs.to_wkt())
        dataset.SetGeoTransform([
            self.x_min_,
            self.pixel_width_,
            0.,
            self.y_max_,
            0.,
            self.pixel_height_,
        ])
        if self.n_bands == 4:
            dataset.GetRasterBand(4).SetMetadataItem('Alpha', '1')

        # Close out the dataset for now. (Reduces likelihood of mem leaks.)
        dataset.FlushCache()
        dataset = None

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

    def handle_out_of_bounds(self, x_off, y_off, x_size, y_size, trim=False):

        # By default we raise an error
        if not trim:

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
                        'of the mosaic'
                    )
            else:
                n_oob = oob.sum()
                if n_oob > 0:
                    raise OutOfBoundsError(
                        'Tried to convert physical to pixels, but '
                        f'{n_oob} of {oob.size} are outside the bounds '
                        'of the mosaic'
                    )

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

    def transform_to_pixel(self, X):

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

    def get_image_with_bounds(self, dataset, x_min, x_max, y_min, y_max):

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

        x_off, y_off, x_size, y_size = self.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )

        return self.get_image(dataset, x_off, y_off, x_size, y_size)

    def save_image_with_bounds(self, dataset, img, x_min, x_max, y_min, y_max):

        x_off, y_off, _, _ = self.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )

        self.save_image(dataset, img, x_off, y_off)

    @staticmethod
    def check_bounds(coords, x_off, y_off, x_size, y_size):

        in_bounds = (
            (x_off <= coords[:, 0])
            & (coords[:, 0] <= x_off + x_size)
            & (y_off <= coords[:, 1])
            & (coords[:, 1] <= y_off + y_size)
        )

        return in_bounds

# class Mosaicker(BaseMosaicker):
# 
#     def __init__(
#         self,
#         config: dict,
#         io_manager: io_management.IOManager = None,
#         crs: Union[str, pyproj.CRS] = 'EPSG:3857',
#         pixel_width: float = None,
#         pixel_height: float = None,
#         dtype: type = np.uint8,
#         n_bands: int = 4,
#         log_keys: list[str] = ['ind', 'return_code'],
#         image_operator: processors.ImageBlender = None,
#     ):
# 
#         # Default settings for file manipulation
#         if io_manager is None:
#             io_manager_options = dict(
#                 filename='mosaic.tiff',
#                 file_exists='error',
#                 aux_files={
#                     'settings': 'settings.yaml',
#                     'log': 'log.csv',
#                     'y_pred': 'y_pred.csv',
#                 },
#                 checkpoint_freq=100,
#                 checkpoint_subdir='checkpoints',
#             )
#             if 'io_manager' in config:
#                 io_manager_options.update(config['io_manager'])
#             io_manager = io_management.IOManager(**io_manager_options)
#         self.io_manager = io_manager
# 
#         self.image_operator = image_operator
# 
#         processor = MosaickerRowTransformer(
#             dtype=dtype,
#             image_operator=image_operator,
#         )
# 
#         super().__init__(
#             processor=processor,
#             out_dir=out_dir,
#             filename=filename,
#             file_exists=file_exists,
#             aux_files=aux_files,
#             checkpoint_freq=checkpoint_freq,
#             checkpoint_subdir=checkpoint_subdir,
#             crs=crs,
#             pixel_width=pixel_width,
#             pixel_height=pixel_height,
#             dtype=dtype,
#             n_bands=n_bands,
#             log_keys=log_keys,
#         )


class SequentialMosaicker(Mosaicker):

    def __init__(
        self,
        io_manager,
        processor,
        mosaicker_train,
        progress_images_subdir: str = 'progress_images',
        save_return_codes: list[str] = [],
        memory_snapshot_freq: int = 10,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        passthrough: Union[bool, list[str]] = True,
        outline: int = 0,
        log_keys: list[str] = ['i', 'ind', 'return_code', 'abs_det_M'],
    ):

        super().__init__(
            io_manager=io_manager,
            processor=processor,
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            passthrough=passthrough,
            outline=outline,
            log_keys=log_keys,
        )

        self.mosaicker_train = mosaicker_train
        self.progress_images_subdir = progress_images_subdir
        self.save_return_codes = save_return_codes
        self.memory_snapshot_freq = memory_snapshot_freq

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        approx_y: pd.DataFrame = None,
        dataset: gdal.Dataset = None,
        i_start: Union[int, str] = 'checkpoint',
    ):

        assert approx_y is not None, \
            'Must pass approx_y.'

        # General fitting
        super().fit(approx_y, dataset=dataset, i_start=i_start)

        # Make a progress images dir
        os.makedirs(
            self.io_manager.output_filepaths['progress_images_dir'],
            exist_ok=True
        )

        # Create the initial mosaic, if not starting from a checkpoint file
        if self.i_start_ == 0:
            dataset = self.io_manager.open_dataset()
            try:
                self.mosaicker_train.fit_transform(X, dataset=dataset)
            except OutOfBoundsError as e:
                raise OutOfBoundsError(
                    "Some of the fitted referenced images are out of bounds. "
                    "Consider increasing the 'padding' in approx_y."
                ) from e

            # Close, to be safe
            dataset.FlushCache()
            dataset = None

            # Save the fit mosaic, pre-prediction
            mosaic_fp = self.io_manager.output_filepaths['mosaic']
            shutil.copy(mosaic_fp, mosaic_fp.replace('.tiff', '_fit.tiff'))

        return self

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        '''Preprocessing required before doing the full loop.
        This should focus on preprocessing that depends on the particular
        image processor (e.g. for mosaickers this includes putting the
        coordinates in the pixel-based frame of the mosaic).

        Parameters
        ----------
        Returns
        -------
        '''

        # Get state of output data
        if self.checkpoint_state_ is None:
            X_t = self.get_search_zone(X)
            X_t = self.transform_to_pixel(X_t)

            # And the logs too
            self.logs = []
        else:
            X_t = self.checkpoint_state_['y_pred']

        # Get the dataset
        resources = {
            'dataset': self.io_manager.open_dataset(),
        }

        return X_t, resources

    def postprocess(self, y_pred, resources):

        # Convert to pixels
        y_pred = self.transform_to_physical(y_pred)
        y_pred['pixel_width'] = self.pixel_width_
        y_pred['pixel_height'] = self.pixel_height_
        y_pred['x_center'] = 0.5 * (y_pred['x_min'] + y_pred['x_max'])
        y_pred['y_center'] = 0.5 * (y_pred['y_min'] + y_pred['y_max'])

        y_pred.to_csv(self.io_manager.output_filepaths['y_pred'])

        # Store log
        log_df = pd.DataFrame(self.logs)
        log_df.to_csv(self.io_manager.output_filepaths['log'])

        # Flush data to disk
        resources['dataset'].FlushCache()
        resources['dataset'] = None

        return y_pred

    def get_search_zone(self, X: pd.DataFrame) -> pd.DataFrame:

        X['x_min'] = X['x_min'] - X['padding']
        X['x_max'] = X['x_max'] + X['padding']
        X['y_min'] = X['y_min'] - X['padding']
        X['y_max'] = X['y_max'] + X['padding']

        return X
