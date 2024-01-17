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

from ..exceptions import OutOfBoundsError
from ..transformers import preprocessors
from . import operators

from .batch import BatchProcessor

from .. import (
    utils, raster
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

        # The transformer for changing between physical and pixel coordinates
        self.transformer_ = raster.RasterCoordinateTransformer()

        # If we have a loaded dataset by this point, get fit params from it
        if dataset is not None:
            self.transformer_.fit(X, dataset=dataset, crs=self.crs)

        # Otherwise, make a new dataset
        else:
            if self.i_start_ != 0:
                raise ValueError(
                    'Creating a new dataset, '
                    'but the starting iteration is not 0. '
                    'If creating a new dataset, should start with i = 0.'
                )
            self.create_containing_dataset(X)
            self.transformer_.input_fit(
                self.x_min_, self.x_max_,
                self.y_min_, self.y_max_,
                self.pixel_width_, self.pixel_height_,
                self.crs,
                self.x_size_, self.y_size_,
            )

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

        X_t = self.transformer_.transform_to_pixel(X)

        # Get the dataset
        resources = {
            'dataset': self.io_manager.open_dataset(),
            'coord_transformer': self.transformer_,
        }

        return X_t, resources

    def postprocess(self, X_t, resources):

        # Close out the dataset
        resources['dataset'].FlushCache()
        resources['dataset'] = None

        return X_t

    # def get_fit_from_dataset(self, dataset):

    #     # Get the dataset bounds
    #     (
    #         (self.x_min_, self.x_max_),
    #         (self.y_min_, self.y_max_),
    #         self.pixel_width_, self.pixel_height_,
    #     ) = raster.get_bounds_from_dataset(
    #         dataset,
    #         self.crs,
    #     )
    #     self.x_size_ = dataset.RasterXSize
    #     self.y_size_ = dataset.RasterYSize

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

        return dataset

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
