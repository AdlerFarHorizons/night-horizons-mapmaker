import copy
import glob
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
from . import processors

from .base import BaseBatchProcesser, BaseRowProcessor

from .. import (
    file_management, preprocessors, utils, raster, metrics
)


class BaseMosaicker(BaseBatchProcesser):
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
        row_processor,
        file_manager,
        out_dir: str,
        filename: str = 'mosaic.tiff',
        file_exists: str = 'error',
        aux_files: dict[str] = {
            'settings': 'settings.yaml',
            'log': 'log.csv',
            'y_pred': 'y_pred.csv',
        },
        checkpoint_freq: int = 100,
        checkpoint_subdir: str = 'checkpoints',
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        dtype: type = np.uint8,
        fill_value: Union[int, float] = None,
        n_bands: int = 4,
        outline: int = 0,
        log_keys: list[str] = ['ind', 'return_code'],
        passthrough: Union[list[str], bool] = False,
    ):

        # Store settings for latter use
        self.row_processor = row_processor
        self.out_dir = out_dir
        self.filename = filename
        self.file_exists = file_exists
        self.aux_files = aux_files
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_subdir = checkpoint_subdir
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.dtype = dtype
        self.fill_value = fill_value
        self.n_bands = n_bands
        self.outline = outline
        self.log_keys = log_keys
        self.passthrough = passthrough

        self.required_columns = ['filepath'] + preprocessors.GEOTRANSFORM_COLS

        # Handles the more-complicated I/O (filetree prep, checkpoints, etc)
        self.file_manager = file_management.MosaicFileManager(
            out_dir=out_dir,
            filename=filename,
            file_exists=file_exists,
            aux_files=aux_files,
            checkpoint_freq=checkpoint_freq,
            checkpoint_subdir=checkpoint_subdir,
        )

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

        # Convert CRS as needed
        # TODO: When making object-creation consistent, address this too.
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # If the dataset was not passed in, load it if possible
        if (
            ((self.file_exists == 'load') and os.path.isfile(self.filepath_))
            or (self.i_start_ != 0)
        ):
            if dataset is not None:
                raise ValueError(
                    'Cannot both pass in a dataset and load a file')
            dataset = self.open_dataset()

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

        # Fit the row processor too
        self.row_processor.fit(self)

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

        X_t = self.transform_to_pixel(X, padding=0)

        # Get the dataset
        resources = {
            'dataset': self.open_dataset(),
        }

        return X_t, resources

    def postprocess(self, X_t, resources):

        # Close out the dataset
        resources['dataset'].FlushCache()
        resources['dataset'] = None

        return X_t

    def score(self, X, y=None, tm_metric=cv2.TM_CCOEFF_NORMED):

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=X['padding'],
        )

        # Limit search regions to within the mosaic.
        # Note that this shouldn't be an issue if the fit is done.
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.handle_out_of_bounds(
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size'],
        )

        # Open the dataset
        dataset = self.open_dataset()

        self.scores_ = []
        for i, fp in enumerate(tqdm.tqdm(X['filepath'], ncols=80)):

            row = X.iloc[i]

            actual_img = utils.load_image(fp, dtype=self.dtype)
            mosaic_img = self.get_image(
                dataset,
                row['x_off'],
                row['y_off'],
                row['x_size'],
                row['y_size'],
            )

            r = metrics.image_to_image_ccoeff(
                actual_img,
                mosaic_img[:, :, :3],
                tm_metric=tm_metric,
            )
            self.scores_.append(r)

        score = np.median(self.scores_)

        return score

    def open_dataset(self):

        return self.file_manager.open_dataset()

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
            self.filepath_,
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
        padding=0,
    ):
        '''
        Parameters
        ----------
        Returns
        -------
        '''

        # Get physical dimensions
        x_imgframe = x_min - self.x_min_ - padding
        y_imgframe = self.y_max_ - y_max - padding
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding

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

    def transform_to_pixel(self, X, padding):

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=padding,
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
#         file_manager: file_management.FileManager = None,
#         crs: Union[str, pyproj.CRS] = 'EPSG:3857',
#         pixel_width: float = None,
#         pixel_height: float = None,
#         dtype: type = np.uint8,
#         n_bands: int = 4,
#         log_keys: list[str] = ['ind', 'return_code'],
#         image_processor: processors.ImageBlender = None,
#     ):
# 
#         # Default settings for file manipulation
#         if file_manager is None:
#             file_manager_options = dict(
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
#             if 'file_manager' in config:
#                 file_manager_options.update(config['file_manager'])
#             file_manager = file_management.FileManager(**file_manager_options)
#         self.file_manager = file_manager
# 
#         self.image_processor = image_processor
# 
#         row_processor = MosaickerRowTransformer(
#             dtype=dtype,
#             image_processor=image_processor,
#         )
# 
#         super().__init__(
#             row_processor=row_processor,
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


class SequentialMosaicker(BaseMosaicker):

    def __init__(
        self,
        out_dir: str,
        filename: str = 'mosaic.tiff',
        file_exists: str = 'error',
        aux_files: dict[str] = {
            'settings': 'settings.yaml',
            'log': 'log.csv',
            'y_pred': 'y_pred.csv',
        },
        checkpoint_freq: int = 100,
        checkpoint_subdir: str = 'checkpoints',
        progress_images_subdir: str = 'progress_images',
        save_return_codes: list[str] = [],
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        passthrough: Union[bool, list[str]] = False,
        outline: int = 0,
        image_joiner: Union[
            processors.ImageJoiner, processors.ImageJoinerQueue
        ] = None,
        log_keys: list[str] = ['i', 'ind', 'return_code', 'abs_det_M'],
        memory_snapshot_freq: int = 10,
    ):

        super().__init__(
            out_dir=out_dir,
            filename=filename,
            file_exists=file_exists,
            aux_files=aux_files,
            checkpoint_freq=checkpoint_freq,
            checkpoint_subdir=checkpoint_subdir,
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
        self.reffed_mosaic = Mosaicker(
            out_dir=out_dir,
            filename=filename,
            file_exists='pass',
            aux_files={'settings': 'settings_initial.yaml'},
            checkpoint_freq=None,
            checkpoint_subdir=checkpoint_subdir,
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            passthrough=passthrough,
            outline=outline,
        )

        self.image_joiner = image_joiner
        self.progress_images_subdir = progress_images_subdir
        self.memory_snapshot_freq = memory_snapshot_freq
        self.save_return_codes = save_return_codes

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
        self.progress_images_subdir_ = os.path.join(
            self.out_dir_, self.progress_images_subdir)
        os.makedirs(self.progress_images_subdir_, exist_ok=True)

        # Create the initial mosaic, if not starting from a checkpoint file
        if self.i_start_ == 0:
            dataset = self.open_dataset()
            self.reffed_mosaic.out_dir = self.out_dir_
            try:
                self.reffed_mosaic.fit_transform(X, dataset=dataset)
            except OutOfBoundsError as e:
                raise OutOfBoundsError(
                    "Some of the fitted referenced images are out of bounds. "
                    "Consider increasing the 'padding' in approx_y."
                ) from e

            # Close, to be safe
            dataset.FlushCache()
            dataset = None

            # Save the fit mosaic, pre-prediction
            shutil.copy(
                self.filepath_,
                self.filepath_.replace('.tiff', '_fit.tiff'),
            )

        return self

    @utils.enable_passthrough
    def predict(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        '''

        Parameters
        ----------
        Returns
        -------
        '''

        self.log = {}

        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Check if fit has been called
        check_is_fitted(self, 'filepath_')

        # Get state of output data
        if self.checkpoint_state_ is None:
            # Set up y-pred
            y_pred = X[preprocessors.GEOTRANSFORM_COLS].copy()
            y_pred[[
                'x_min', 'x_max',
                'y_min', 'y_max',
                'pixel_width', 'pixel_height',
                'x_size', 'y_size',
                'x_center', 'y_center',
                'x_off', 'y_off',
            ]] = np.nan
            y_pred['return_code'] = 'TBD'

            # And the logs too
            self.logs = []
        else:
            y_pred = self.checkpoint_state_['y_pred']
            self.logs = self.checkpoint_state_['logs']

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=X['padding'],
        )

        # Limit search regions to within the mosaic.
        # Note that this shouldn't be an issue if the fit is done correctly.
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.handle_out_of_bounds(
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size'],
            trim=True,
        )

        dataset = self.open_dataset()

        # Start memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.start()
            start = tracemalloc.take_snapshot()
            self.log['starting_snapshot'] = start

        for i, ind in enumerate(tqdm.tqdm(X.index, ncols=80)):

            if i < self.i_start_:
                continue

            row = X.loc[ind]

            if 'iter_duration' in self.log_keys:
                start = time.time()
            return_code, results, log = self.incorporate_image(
                dataset,
                row,
            )
            # Time it
            if 'iter_duration' in self.log_keys:
                iter_duration = time.time() - start

            if return_code == 'success':
                # Update y_pred
                y_pred.loc[ind, ['x_off', 'y_off', 'x_size', 'y_size']] = [
                    results['x_off'], results['y_off'],
                    results['x_size'], results['y_size']
                ]

            # Snapshot the memory usage
            if 'snapshot' in self.log_keys:
                if i % self.memory_snapshot_freq == 0:
                    log['snapshot'] = tracemalloc.take_snapshot()

            # Store metadata
            y_pred.loc[ind, 'return_code'] = return_code
            log = self.update_log(locals(), target=log)
            self.logs.append(log)

            # Checkpoint (only saves at particular `i` values)
            dataset = self.file_manager.save_to_checkpoint(
                i,
                dataset=dataset,
                y_pred=y_pred,
                logs=self.logs,
            )

        # Convert to pixels
        (
            y_pred['x_min'], y_pred['x_max'],
            y_pred['y_min'], y_pred['y_max'],
        ) = self.pixel_to_physical(
            y_pred['x_off'], y_pred['y_off'],
            y_pred['x_size'], y_pred['y_size']
        )
        y_pred['pixel_width'] = self.pixel_width_
        y_pred['pixel_height'] = self.pixel_height_
        y_pred['x_center'] = 0.5 * (y_pred['x_min'] + y_pred['x_max'])
        y_pred['y_center'] = 0.5 * (y_pred['y_min'] + y_pred['y_max'])

        # Flush data to disk
        dataset.FlushCache()
        dataset = None
        y_pred.to_csv(self.file_manager.aux_filepaths_['y_pred'])

        # Store log
        log_df = pd.DataFrame(self.logs)
        log_df.to_csv(self.file_manager.aux_filepaths_['log'])

        # Stop memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.stop()

        return y_pred

    def incorporate_image(
        self,
        dataset,
        row: pd.Series,
    ):

        results = {}
        log = {}

        # Get image location
        x_off = row['x_off']
        y_off = row['y_off']
        x_size = row['x_size']
        y_size = row['y_size']

        # Get dst features
        # TODO: When dst pts are provided, this step could be made faster by
        #       not loading dst_img at this time.
        dst_img = self.get_image(dataset, x_off, y_off, x_size, y_size)

        # Check what's in bounds, exit if nothing
        if dst_img.sum() == 0:
            log = self.update_log(locals(), target=log)
            return 'out_of_bounds', results, log

        # Get src image
        src_img = utils.load_image(
            row['filepath'],
            dtype=self.dtype,
        )

        # Main function
        # TODO: Return codes may not actually be particularly Pythonic.
        #    However, we need to track *how* the failures happened somehow,
        #    so we need some sort of flag, which is basically a return code.
        #    That said, there may be a better alternative to this.
        return_code, result, image_joiner_log = self.image_joiner.process(
            src_img, dst_img)
        log = self.update_log(image_joiner_log, target=log)

        # TODO: Clean this up
        if return_code == 'success':
            # Store the image
            self.save_image(dataset, result['blended_img'], x_off, y_off)

        # Save failed images for later debugging
        # TODO: Currently the format of the saved images is a little weird.
        if (
            (self.progress_images_subdir_ is not None)
            and (return_code in self.save_return_codes)
        ):
            n_tests_existing = len(glob.glob(os.path.join(
                self.progress_images_subdir_, '*_dst.tiff')))
            dst_fp = os.path.join(
                self.progress_images_subdir_,
                f'{n_tests_existing:06d}_dst.tiff'
            )
            src_fp = os.path.join(
                self.progress_images_subdir_,
                f'{n_tests_existing:06d}_src.tiff'
            )

            cv2.imwrite(src_fp, src_img[:, :, ::-1])
            cv2.imwrite(dst_fp, dst_img[:, :, ::-1])

            if 'blended_img' in result:
                blended_fp = os.path.join(
                    self.progress_images_subdir_,
                    f'{n_tests_existing:06d}_blended.tiff'
                )
                cv2.imwrite(blended_fp, result['blended_img'][:, :, ::-1])

        # Log
        log = self.update_log(locals(), target=log)

        return return_code, results, log


class MosaickerRowTransformer(BaseRowProcessor):

    def __init__(self, image_processor=None, dtype: type = np.uint8):

        if image_processor is None:
            self.image_processor = processors.ImageBlender()
        else:
            self.image_processor = image_processor

        self.dtype = dtype

    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:

        src_img = utils.load_image(
            row['filepath'],
            dtype=self.dtype,
        )

        return {'image': src_img}

    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:

        dst_img = self.get_image_from_dataset(
            resources['dataset'],
            row['x_off'],
            row['y_off'],
            row['x_size'],
            row['y_size'],
        )

        return {'image': dst_img}

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:

        # Combine the images
        # TODO: image_processor is more-general,
        #       but image_blender is more descriptive
        results = self.image_processor.process(
            src['image'],
            dst['image'],
        )
        self.update_log(self.image_processor.log)

        return {'blended_image': results['blended_image']}

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        result: dict,
    ):

        # Store the image
        self.save_image_to_dataset(
            resources['dataset'],
            result['blended_image'],
            row['x_off'],
            row['y_off'],
        )

    ###########################################################################
    # Auxillary functions below

    def get_image_from_dataset(self, dataset, x_off, y_off, x_size, y_size):

        assert x_off >= 0, 'x_off cannot be less than 0'
        assert x_off + x_size <= self.x_size_, \
            'x_off + x_size cannot be greater than self.x_size_'
        assert y_off >= 0, 'y_off cannot be less than 0'
        assert y_off + y_size <= self.y_size_, \
            'y_off + y_size cannot be greater than self.y_size_'

        # Note that we cast the input as int, in case we the input was numpy
        # integers instead of python integers.
        img = dataset.ReadAsArray(
            xoff=int(x_off),
            yoff=int(y_off),
            xsize=int(x_size),
            ysize=int(y_size),
        )
        img = img.transpose(1, 2, 0)

        return img

    def save_image_to_dataset(self, dataset, img, x_off, y_off):

        img_to_save = img.transpose(2, 0, 1)
        dataset.WriteArray(
            img_to_save,
            xoff=int(x_off),
            yoff=int(y_off),
        )
