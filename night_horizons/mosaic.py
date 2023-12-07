import glob
import os
import shutil
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
        file_exists: str = 'error',
        checkpoint_freq: int = 100,
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
        debug_mode: bool = False,
        log_keys: list[str] = [],
        value_exists: str = 'append',
    ):
        self.filepath = filepath
        self.file_exists = file_exists
        self.checkpoint_freq = checkpoint_freq
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

        super().__init__(debug_mode, log_keys, value_exists)

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
            Returns self.
        '''

        # Flexible file-handling. Maybe overkill?
        if os.path.isfile(self.filepath):
            if self.file_exists == 'error':
                raise FileExistsError('File already exists at destination.')
            elif self.file_exists == 'pass':
                pass
            elif self.file_exists == 'overwrite':
                os.remove(self.filepath)
            elif self.file_exists == 'load':
                if dataset is not None:
                    raise ValueError(
                        'Cannot both pass in a dataset and load a file')
                self.dataset_ = gdal.Open(self.filepath, gdal.GA_Update)
            else:
                raise ValueError(
                    'Unrecognized value for filepath, '
                    f'filepath={self.filepath}'
                )

        if dataset is not None:
            self.dataset_ = dataset

        # Load the dataset if it already exists
        if hasattr(self, 'dataset_'):

            # Get the dataset bounds
            (
                (self.x_min_, self.x_max_),
                (self.y_min_, self.y_max_),
                self.pixel_width_, self.pixel_height_
            ) = raster.get_bounds_from_dataset(
                self.dataset_,
                self.crs,
            )
            self.x_size_ = self.dataset_.RasterXSize
            self.y_size_ = self.dataset_.RasterYSize

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
        self.dataset_ = driver.Create(
            self.filepath,
            xsize=self.x_size_,
            ysize=self.y_size_,
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

        self.scores_ = []
        for i, fp in enumerate(tqdm.tqdm(X['filepath'], ncols=80)):

            row = X.iloc[i]

            actual_img = utils.load_image(fp, dtype=self.dtype)
            mosaic_img = self.get_image(
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

    def close(self):

        self.dataset_.FlushCache()
        self.dataset_ = None

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
            x_off[x_off + x_size >= self.x_size_] = self.x_size_ - 1 - x_off
            y_off[y_off + y_size >= self.y_size_] = self.y_size_ - 1 - y_off

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

    def get_image(self, x_off, y_off, x_size, y_size):

        # Note that we cast the input as int, in case we the input was numpy
        # integers instead of python integers.
        img = self.dataset_.ReadAsArray(
            xoff=int(x_off),
            yoff=int(y_off),
            xsize=int(x_size),
            ysize=int(y_size),
        )
        return img.transpose(1, 2, 0)

    def save_image(self, img, x_off, y_off):

        img_to_save = img.transpose(2, 0, 1)
        self.dataset_.WriteArray(
            img_to_save,
            xoff=int(x_off),
            yoff=int(y_off),
        )

    def get_image_with_bounds(self, x_min, x_max, y_min, y_max):

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

        return self.get_image(x_off, y_off, x_size, y_size)

    def save_image_with_bounds(self, img, x_min, x_max, y_min, y_max):

        x_off, y_off, _, _ = self.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )

        self.save_image(img, x_off, y_off)

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
        check_is_fitted(self, 'dataset_')

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
            self.save_image(blended_img, row['x_off'], row['y_off'])

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
        file_exists: str = 'overwrite',
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
        debug_mode: bool = True,
        log_keys: list[str] = ['abs_det_M', 'i', 'ind'],
        value_exists: str = 'append',
        bad_images_dir: str = None,
        save_return_codes: list[str] = [],
    ):

        super().__init__(
            filepath=filepath,
            file_exists=file_exists,
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
            debug_mode=debug_mode,
            log_keys=log_keys,
            value_exists=value_exists,
        )
        self.reffed_mosaic = ReferencedMosaic(
            filepath=filepath,
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
            debug_mode=False,
        )

        self.image_joiner = image_joiner
        self.feature_mode = feature_mode
        self.bad_images_dir = bad_images_dir
        self.save_return_codes = save_return_codes

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        approx_y: pd.DataFrame = None,
        dataset: gdal.Dataset = None,
    ):

        assert approx_y is not None, \
            'Must pass approx_y.'

        # Create the dataset
        super().fit(approx_y, dataset=dataset)

        # Add the existing mosaic
        self.reffed_mosaic.fit_transform(X, dataset=self.dataset_)

    @utils.enable_passthrough
    def predict(
        self,
        X: pd.DataFrame,
        y=None,
        iteration_indices: np.ndarray[int] = None,
    ):
        ''' TODO: Deprecate iteration_indices. Just have the user order their
        dataframe prior to input.

        Parameters
        ----------
        Returns
        -------
        '''
        # Useful for debugging
        self.log = {}

        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        if iteration_indices is None:
            iteration_indices = self.calc_iteration_indices(X)

        # Check if fit had been called
        check_is_fitted(self, 'dataset_')

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

        # Convert to pixels
        (
            X['x_off'], X['y_off'],
            X['x_size'], X['y_size']
        ) = self.physical_to_pixel(
            X['x_min'], X['x_max'],
            X['y_min'], X['y_max'],
            padding=self.padding * X['spatial_error'],
        )

        # Get the features for the existing mosaic
        if self.feature_mode == 'store':
            dst_img = self.get_image(0, 0, self.x_size_, self.y_size_)
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
            iterable = tqdm.tqdm(iteration_indices, ncols=80)
        else:
            iterable = iteration_indices
        for i, ind in enumerate(iterable):

            if i % self.checkpoint_freq:
                self.dataset_.FlushCache()

            row = X.loc[ind]

            return_code, results = self.incorporate_image(
                row,
                dsframe_dst_pts,
                dsframe_dst_des,
            )

            # Store return code and continue, if failed
            if return_code != 'success':
                continue

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

        return y_pred[preprocess.GEOTRANSFORM_COLS]

    def incorporate_image(
        self,
        row: pd.Series,
        dsframe_dst_pts: np.ndarray,
        dsframe_dst_des: np.ndarray,
    ):

        results = {}

        # Get image location
        x_off = row['x_off']
        y_off = row['y_off']
        x_size = row['x_size']
        y_size = row['y_size']

        # Get dst features
        # TODO: When dst pts are provided, this step could be made faster by
        #       not loading dst_img at this time.
        dst_img = self.get_image(x_off, y_off, x_size, y_size)
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
                self.update_log(locals())
                return 'out_of_bounds', results

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
        return_code, result, log = self.image_joiner.join(
            src_img, dst_img)

        # TODO: Clean this up
        if return_code == 'success':
            # Store the image
            self.save_image(result['blended_img'], x_off, y_off)

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

                raster.Image(dst_img).save(dst_fp)
                shutil.copy(row['filepath'], src_fp)

        # Log
        # TODO: This is such a fragile way to log.
        #       It requires careful knowledge of the state of the log,
        #       and whether or not it has been called already.
        log.update(locals())
        self.update_log(log)

        return return_code, results

    def close(self):

        self.reffed_mosaic.close()
        super().close()
