import gc
import glob
import inspect
import os
import pickle
import re
import shutil
import tracemalloc
from typing import Tuple, Union
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
import yaml

from . import utils, raster, preprocess, features, metrics


class Mosaic(utils.LoggerMixin, TransformerMixin, BaseEstimator):
    '''Assemble a mosaic from georeferenced images.

    TODO: padding is a parameter right now, but in reality it's image
    dependent, so it would be nice to have it as a column instead.

    TODO: filepath is a data-dependent parameter, so it really should be
    called at the time of the fit.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        filepath: str,
        y_pred_filepath_ext: str = '_y_pred.csv',
        settings_filepath_ext: str = '_settings.yaml',
        log_filepath_ext: str = '_log.csv',
        file_exists: str = 'error',
        save_aux_files: bool = True,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        padding: float = 0.,
        dataset_padding: float = 0.,
        passthrough: Union[bool, list[str]] = False,
        outline: int = 0,
        verbose: bool = True,
        log_keys: list[str] = ['ind', 'return_code'],
    ):
        self.filepath = filepath
        self.y_pred_filepath_ext = y_pred_filepath_ext
        self.settings_filepath_ext = settings_filepath_ext
        self.log_filepath_ext = log_filepath_ext
        self.file_exists = file_exists
        self.save_aux_files = save_aux_files
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fill_value = fill_value
        self.dtype = dtype
        self.n_bands = n_bands
        self.padding = padding
        self.dataset_padding = dataset_padding
        self.passthrough = passthrough
        self.outline = outline
        self.verbose = verbose

        self.required_columns = ['filepath'] + preprocess.GEOTRANSFORM_COLS

        super().__init__(log_keys)

    @utils.enable_passthrough
    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        dataset: gdal.Dataset = None,
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
            Returns self
        '''

        # Filepaths
        self.filepath_ = self.filepath
        base, ext = os.path.splitext(self.filepath_)
        self.y_pred_filepath_ = base + self.y_pred_filepath_ext
        self.settings_filepath_ = base + self.settings_filepath_ext
        self.log_filepath_ = base + self.log_filepath_ext

        # Flexible file-handling. Maybe overkill?
        if os.path.isfile(self.filepath_):

            # Standard, simple options
            if self.file_exists == 'error':
                raise FileExistsError('File already exists at destination.')
            elif self.file_exists == 'pass':
                pass
            elif self.file_exists == 'overwrite':
                os.remove(self.filepath_)
            elif self.file_exists == 'load':
                if dataset is not None:
                    raise ValueError(
                        'Cannot both pass in a dataset and load a file')
                dataset = gdal.Open(self.filepath, gdal.GA_Update)

            # Create a new file with a new number appended
            elif self.file_exists == 'new':
                base, ext = os.path.splitext(self.filepath)
                new_fp_format = base + '_v{:03d}' + ext
                self.filepath_ = new_fp_format.format(0)
                i = 0
                while os.path.isfile(self.filepath_):
                    self.filepath_ = new_fp_format.format(i)
                    i += 1

                # Change auxiliary files too
                base, ext = os.path.splitext(self.filepath_)
                self.y_pred_filepath_ = base + self.y_pred_filepath_ext
                self.settings_filepath_ = base + self.settings_filepath_ext
                self.log_filepath_ = base + self.log_filepath_ext

            else:
                raise ValueError(
                    'Unrecognized value for filepath, '
                    f'filepath={self.filepath_}'
                )

        # We always remove the auxiliary files, if they already exist
        for fp in [self.y_pred_filepath_, self.settings_filepath_]:
            if os.path.isfile(fp):
                os.remove(fp)

        self.save_settings()

        # Use the dataset if it already exists
        if dataset is not None:

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

            return self

        # Check the input is good.
        # TODO: For functions decorated by enable_passthrough this is
        #       degenerate
        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Convert CRS as needed
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Get bounds
        self.x_min_ = X['x_min'].min() - self.dataset_padding
        self.x_max_ = X['x_max'].max() + self.dataset_padding
        self.y_min_ = X['y_min'].min() - self.dataset_padding
        self.y_max_ = X['y_max'].max() + self.dataset_padding

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

        return self

    def score(self, X, y=None, tm_metric=cv2.TM_CCOEFF_NORMED):

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=self.padding,
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

        return gdal.Open(self.filepath_, gdal.GA_Update)

    def save_settings(self):

        fullargspec = inspect.getfullargspec(type(self))
        settings = {}
        for setting in fullargspec.args:
            if setting == 'self':
                continue
            value = getattr(self, setting)
            try:
                pickle.dumps(value)
            except TypeError:
                value = 'no string repr'
            settings[setting] = value
        with open(self.settings_filepath_, 'w') as file:
            yaml.dump(settings, file)

    def calc_iteration_indices(self, X):

        return X.index

        # TODO: restore decent defaults?
        d_to_center = np.sqrt(
            (X['x_center'] - self.central_coords_[0])**2.
            + (X['y_center'] - self.central_coords_[1])**2.
        )
        iteration_indices = d_to_center.sort_values().index

        return iteration_indices

    def physical_to_pixel(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
        padding=0
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

            # Handle out-of-bounds
            x_off[x_off < 0] = 0
            y_off[y_off < 0] = 0
            x_size[x_off + x_size > self.x_size_] = self.x_size_ - x_off
            y_size[y_off + y_size > self.y_size_] = self.y_size_ - y_off

        # When we're passing in single values.
        except TypeError:
            # Change dtypes
            x_off = int(x_off)
            y_off = int(y_off)
            x_size = int(x_size)
            y_size = int(y_size)

            # Handle out-of-bounds
            if x_off < 0:
                x_off = 0
            elif x_off + x_size >= self.x_size_:
                x_off = self.x_size_ - 1 - x_off
            if y_off < 0:
                y_off = 0
            elif y_off + y_size >= self.y_size_:
                y_off = self.y_size_ - 1 - y_off

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

    def get_image(self, dataset, x_off, y_off, x_size, y_size):

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
        return img.transpose(1, 2, 0)

    def save_image(self, dataset, img, x_off, y_off):

        img_to_save = img.transpose(2, 0, 1)
        dataset.WriteArray(
            img_to_save,
            xoff=int(x_off),
            yoff=int(y_off),
        )

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


class ReferencedMosaic(Mosaic):

    @utils.enable_passthrough
    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Check if fit had been called
        check_is_fitted(self, 'filepath_')

        # DEBUG: Some x_offs can be negative, probably due to padding
        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=self.padding,
        )

        # Get the dataset
        dataset = self.open_dataset()

        # If verbose, add a progress bar.
        if self.verbose:
            iterable = tqdm.tqdm(X['filepath'], ncols=80)
        else:
            iterable = X['filepath']
        for i, fp in enumerate(iterable):

            row = X.iloc[i]

            # Get data
            src_img = utils.load_image(
                fp,
                dtype=self.dtype,
            )
            dst_img = self.get_image(
                dataset,
                row['x_off'],
                row['y_off'],
                row['x_size'],
                row['y_size'],
            )

            # Resize the source image
            src_img_resized = cv2.resize(
                src_img,
                (dst_img.shape[1], dst_img.shape[0])
            )

            # Combine the images
            blended_img = utils.blend_images(
                src_img=src_img_resized,
                dst_img=dst_img,
                fill_value=self.fill_value,
                outline=self.outline,
            )

            # Store the image
            self.save_image(dataset, blended_img, row['x_off'], row['y_off'])

    def predict(
        self,
        X: pd.DataFrame,
    ):
        '''Transform and predict mean the same thing here.
        Transform is the appropriate term when we're changing the referenced
        images into a mosaic, and are assuming the mosaic as the ground truth.
        Predict is the appropriate term when we're assessing the accuracy of
        the created mosaic.
        '''

        return self.transform(X)


class LessReferencedMosaic(Mosaic):

    def __init__(
        self,
        filepath: str,
        y_pred_filepath_ext: str = '_y_pred.csv',
        settings_filepath_ext: str = '_settings.yaml',
        log_filepath_ext: str = '_log.csv',
        file_exists: str = 'new',
        checkpoint_freq: int = 100,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        padding: float = 0.,
        dataset_padding: float = 5000.,
        passthrough: Union[bool, list[str]] = False,
        outline: int = 0,
        verbose: bool = True,
        image_joiner: Union[
            features.ImageJoiner, features.ImageJoinerQueue
        ] = None,
        feature_mode: str = 'recompute',
        log_keys: list[str] = ['i', 'ind', 'return_code', 'abs_det_M'],
        bad_images_dir: str = None,
        memory_snapshot_freq: int = 10,
        save_return_codes: list[str] = [],
    ):

        super().__init__(
            filepath=filepath,
            file_exists=file_exists,
            y_pred_filepath_ext=y_pred_filepath_ext,
            settings_filepath_ext=settings_filepath_ext,
            log_filepath_ext=log_filepath_ext,
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            padding=padding,
            dataset_padding=dataset_padding,
            passthrough=passthrough,
            outline=outline,
            verbose=verbose,
            log_keys=log_keys,
        )
        self.reffed_mosaic = ReferencedMosaic(
            filepath=filepath,
            settings_filepath_ext='_initial' + settings_filepath_ext,
            y_pred_filepath_ext='_initial' + y_pred_filepath_ext,
            file_exists='pass',
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            padding=0.,
            dataset_padding=0.,
            passthrough=passthrough,
            outline=outline,
            verbose=verbose,
        )

        self.checkpoint_freq = checkpoint_freq
        self.image_joiner = image_joiner
        self.feature_mode = feature_mode
        self.bad_images_dir = bad_images_dir
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

        # Start with a fresh log
        self.log = {}
        self.logs = []

        assert approx_y is not None, \
            'Must pass approx_y.'

        # Create the dataset
        super().fit(approx_y, dataset=dataset)

        # TODO: Change to providing a directory directly, instead of inferring
        #       from the filepath?
        if i_start == 'checkpoint':

            # Determine what to look for, for checkpoint files
            checkpoint_dir, filename = os.path.split(self.filepath_)
            checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            base, ext = os.path.splitext(filename)
            i_tag = r'_i(\d+)'
            checkpoint_pattern = base + i_tag + ext
            pattern = re.compile(checkpoint_pattern)

            # Look for checkpoint files
            i_start = -1
            j_filename = None
            possible_files = os.listdir(checkpoint_dir)
            for j, filename in enumerate(possible_files):
                match = pattern.search(filename)
                if not match:
                    continue

                number = int(match.group(1))
                if number > i_start:
                    i_start = number
                    j_filename = j

            if i_start != -1:

                print(
                    'Found checkpoint file. '
                    f'Will fast forward to i={i_start + 1}'
                )

                # Copy over dataset
                filename = possible_files[j_filename]
                filepath = os.path.join(checkpoint_dir, filename)
                shutil.copy(filepath, self.filepath_)

                # Open the log
                base, ext = os.path.splitext(self.filepath_)
                log_filepath = base + self.log_filepath_ext
                log_df = pd.read_csv(log_filepath)
                log_df = log_df[self.log_keys]

                # Format the stored logs
                for i, ind in enumerate(log_df.index):
                    if i > i_start:
                        break
                    log = dict(log_df.loc[ind])
                    self.logs.append(log)

            # We don't want to start on the same loop that was saved, but the
            # one after
            i_start += 1
        self.i_start_ = i_start

        # Create the initial mosaic, if not starting from a checkpoint file
        if self.i_start_ == 0:
            dataset = self.open_dataset()
            self.reffed_mosaic.filepath = self.filepath_
            self.reffed_mosaic.fit_transform(X, dataset=dataset)

    @utils.enable_passthrough
    def predict(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        ''' TODO: Deprecate iteration_indices. Just have the user order their
        dataframe prior to input.

        Parameters
        ----------
        Returns
        -------
        '''

        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Check if fit had been called
        check_is_fitted(self, 'filepath_')

        # Set up y_pred
        y_pred = X[preprocess.GEOTRANSFORM_COLS].copy()
        y_pred[[
            'x_min', 'x_max',
            'y_min', 'y_max',
            'pixel_width', 'pixel_height',
            'x_size', 'y_size',
            'x_center', 'y_center',
            'x_off', 'y_off',
        ]] = np.nan
        y_pred['return_code'] = 'TBD'

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=self.padding * X['spatial_error'],
        )

        dataset = self.open_dataset()

        # Get the features for the existing mosaic
        if self.feature_mode == 'store':
            dst_img = self.get_image(dataset, 0, 0, self.x_size_, self.y_size_)
            dsframe_dst_kps, dsframe_dst_des = \
                self.feature_detector_.detectAndCompute(dst_img, None)
            dsframe_dst_pts = cv2.KeyPoint_convert(dsframe_dst_kps)
        # Or indicate we will not be passing those in.
        elif self.feature_mode == 'recompute':
            dsframe_dst_pts = None
            dsframe_dst_des = None
        else:
            raise ValueError(
                f'feature_mode = {self.feature_mode} is not a valid option. '
                "Valid options are ['store', 'recompute']."
            )

        # If verbose, add a progress bar.
        if self.verbose:
            iterable = tqdm.tqdm(X.index, ncols=80)
        else:
            iterable = X.index

        # Start memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.start()
            start = tracemalloc.take_snapshot()
            self.log['starting_snapshot'] = start

        for i, ind in enumerate(iterable):

            if i < self.i_start_:
                continue

            row = X.loc[ind]

            return_code, results, log = self.incorporate_image(
                dataset,
                row,
                dsframe_dst_pts,
                dsframe_dst_des,
            )

            if return_code == 'success':

                # Store the transformed points for the next loop
                if self.feature_mode == 'store':
                    dsframe_dst_pts = np.append(
                        dsframe_dst_pts,
                        results['dsframe_src_pts'],
                        axis=0
                    )
                    dsframe_dst_des = np.append(
                        dsframe_dst_des,
                        results['src_des'],
                        axis=0
                    )

                # Update y_pred
                y_pred.loc[ind, ['x_off', 'y_off', 'x_size', 'y_size']] = [
                    results['x_off'], results['y_off'],
                    results['x_size'], results['y_size']
                ]

            # Snapshot the memory usage
            if 'snapshot' in self.log_keys:
                if i % self.memory_snapshot_freq == 0:
                    log['snapshot'] = tracemalloc.take_snapshot()

            # Checkpoint
            if (i % self.checkpoint_freq == 0) and (i != 0):

                # Flush data to disk
                dataset.FlushCache()
                dataset = None
                y_pred.to_csv(self.y_pred_filepath_)

                # TODO: Clean up filepath manipulations
                # Make checkpoint file by copying dataset
                checkpoint_dir, filename = os.path.split(self.filepath_)
                base, ext = os.path.splitext(filename)
                i_tag = f'_i{i:06d}'
                filename = base + i_tag + ext
                checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
                checkpoint_fp = os.path.join(checkpoint_dir, filename)
                shutil.copy(self.filepath_, checkpoint_fp)

                # Store log
                log_df = pd.DataFrame(self.logs)
                log_df.to_csv(self.log_filepath_)

                # Re-open dataset
                dataset = self.open_dataset()

            # Store metadata
            y_pred.loc[ind, 'return_code'] = return_code
            log = self.update_log(locals(), target=log)
            self.logs.append(log)

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
        y_pred.to_csv(self.y_pred_filepath_)

        # Store log
        log_df = pd.DataFrame(self.logs)
        log_df.to_csv(self.log_filepath_)

        # Stop memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.stop()

        return y_pred

    def incorporate_image(
        self,
        dataset,
        row: pd.Series,
        dsframe_dst_pts: np.ndarray,
        dsframe_dst_des: np.ndarray,
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
        if self.feature_mode == 'store':
            raise NotImplementedError('Removed this functionality for now.')

            # Check what's in bounds, exit if nothing
            in_bounds = self.check_bounds(
                dsframe_dst_pts,
                x_off, y_off, x_size, y_size
            )
            if in_bounds.sum() == 0:
                debug_log = self.log_locals(locals())
                return 'out_of_bounds', results, debug_log

            # Get pts in the local frame
            dst_pts = dsframe_dst_pts[in_bounds] - np.array([x_off, y_off])
            dst_kp = cv2.KeyPoint_convert(dst_pts)
            dst_des = dsframe_dst_des[in_bounds]
        else:
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
        return_code, result, image_joiner_log = self.image_joiner.join(
            src_img, dst_img)
        log = self.update_log(image_joiner_log, target=log)

        # TODO: Clean this up
        if return_code == 'success':
            # Store the image
            self.save_image(dataset, result['blended_img'], x_off, y_off)

            # Auxiliary: Convert to the dataset frame
            src_pts = cv2.KeyPoint_convert(result['src_kp'])
            dsframe_src_pts = cv2.perspectiveTransform(
                src_pts.reshape(-1, 1, 2),
                result['M'],
            ).reshape(-1, 2)
            dsframe_src_pts += np.array([x_off, y_off])
            results['dsframe_src_pts'] = dsframe_src_pts

            # Auxiliary: Convert bounding box (needed for georeferencing)
            (
                results['x_off'], results['y_off'],
                results['x_size'], results['y_size']
            ) = utils.warp_bounds(src_img, result['M'])
            results['x_off'] += x_off
            results['y_off'] += y_off

        # Save failed images for later debugging
        elif return_code in self.save_return_codes:
            if self.bad_images_dir is not None:
                n_tests_existing = len(glob.glob(os.path.join(
                    self.bad_images_dir, 'dst_*.tiff')))
                dst_fp = os.path.join(
                    self.bad_images_dir, f'dst_{n_tests_existing:03d}.tiff')
                src_fp = os.path.join(
                    self.bad_images_dir, f'src_{n_tests_existing:03d}.tiff')

                cv2.imwrite(src_fp, src_img[:, :, ::-1])
                cv2.imwrite(dst_fp, dst_img[:, :, ::-1])

        # Log
        log = self.update_log(locals(), target=log)

        return return_code, results, log
