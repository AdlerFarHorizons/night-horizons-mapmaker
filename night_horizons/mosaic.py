import glob
import os
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

from . import preprocess, raster, metrics, utils


ERROR_CODE_MAP = {
    0: 'Success',
    1: 'Extreme homography (large determinant)',
    2: 'OpenCV error',
    3: 'No in-bounds image data to match with',
}


class Mosaic(TransformerMixin, BaseEstimator):
    '''Assemble a mosaic from georeferenced images.

    TODO: padding is a parameter right now, but in reality it's image
    dependent, so it would be nice to have it as a column instead.

    TODO: filepath is a data-dependent parameter, so it really should be
    called at the time of the fit.

    TODO: convert the coordinates to camera frame as part of a separate loop,
        not multiple times per image loop

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        filepath: str,
        file_exists: str = 'error',
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
        debug: bool = False,
    ):
        self.filepath = filepath
        self.file_exists = file_exists
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
        self.debug = debug

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

        # Change dtypes
        try:
            x_off = x_off.astype(int)
            y_off = y_off.astype(int)
            x_size = x_size.astype(int)
            y_size = y_size.astype(int)
        except TypeError:
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

        return self.get_image(self, x_off, y_off, x_size, y_size)

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
        homography_det_min=0.5,
        feature_detector: str = 'AKAZE',
        feature_detector_kwargs: dict = {},
        feature_matcher: str = 'BFMatcher',
        feature_matcher_kwargs: dict = {},
        feature_mode: str = 'recompute',
        log_keys: list[str] = ['abs_det_M'],
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
        )

        self.homography_det_min = homography_det_min

        self.feature_detector = feature_detector
        self.feature_detector_kwargs = feature_detector_kwargs
        self.feature_matcher = feature_matcher
        self.feature_matcher_kwargs = feature_matcher_kwargs
        self.feature_mode = feature_mode
        self.log_keys = log_keys

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

        # DEBUG
        # import pdb; pdb.set_trace()

        # Add the existing mosaic
        self.reffed_mosaic.fit_transform(X, dataset=self.dataset_)

        # Make the feature detector and matcher
        # We do the somewhat circuitous rout of passing in the name of
        # the feature detector and the arguments separately, as opposed to
        # passing in a class. This is because cv2 classes can't be pickled.
        constructor = getattr(cv2, f'{self.feature_detector}_create')
        self.feature_detector_ = constructor(**self.feature_detector_kwargs)
        constructor = getattr(cv2, self.feature_matcher)
        self.feature_matcher_ = constructor(**self.feature_matcher_kwargs)

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

        # Loop through and include
        self.log_ = {
            'return_codes': [],
        }
        for log_key in self.log_keys:
            self.log_[log_key] = []
        # If verbose, add a progress bar.
        if self.verbose:
            iterable = tqdm.tqdm(iteration_indices, ncols=80)
        else:
            iterable = iteration_indices
        for i, ind in enumerate(iterable):

            self.log_['last_i'] = i
            self.log_['last_ind'] = ind

            row = X.loc[ind]

            try:
                return_code, results, log_i = self.incorporate_image(
                    row,
                    dsframe_dst_pts,
                    dsframe_dst_des,
                )
            except cv2.error:
                log_i = {}
                return_code = 2

            # Logging
            for log_key in self.log_keys:
                if log_key in log_i:
                    self.log_[log_key].append(log_i[log_key])
                else:
                    self.log_[log_key].append(np.nan)

            # Store return code and continue, if failed
            self.log_['return_codes'].append(return_code)
            if return_code != 0:
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

        self.log_['dsframe_dst_pts'] = dsframe_dst_pts
        self.log_['dsframe_dst_des'] = dsframe_dst_des

        return y_pred[preprocess.GEOTRANSFORM_COLS]

    def incorporate_image(
        self,
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
        dst_img = self.get_image(x_off, y_off, x_size, y_size)
        if (dsframe_dst_pts is not None) and (dsframe_dst_des is not None):

            # Check what's in bounds, exit if nothing
            in_bounds = self.check_bounds(
                dsframe_dst_pts,
                x_off, y_off, x_size, y_size
            )
            if in_bounds.sum() == 0:
                if self.debug:
                    log = {
                        log_key: locals()[log_key]
                        for log_key in self.log_keys
                        if log_key in locals()
                    }
                return 3, results, log

            # Get pts in the local frame
            dst_pts = dsframe_dst_pts[in_bounds] - np.array([x_off, y_off])
            dst_kp = cv2.KeyPoint_convert(dst_pts)
            dst_des = dsframe_dst_des[in_bounds]
        else:
            # Check what's in bounds, exit if nothing
            if dst_img.sum() == 0:
                if self.debug:
                    log = {
                        log_key: locals()[log_key]
                        for log_key in self.log_keys
                        if log_key in locals()
                    }
                return 3, results, log

            # Get the pts
            dst_kp, dst_des = self.feature_detector_.detectAndCompute(
                dst_img, None)

        # Get src features
        src_img = utils.load_image(
            row['filepath'],
            dtype=self.dtype,
        )
        src_kp, src_des = self.feature_detector_.detectAndCompute(
            src_img, None)
        results['src_des'] = src_des

        # Feature matching
        M = utils.calc_warp_transform(
            src_kp,
            src_des,
            dst_kp,
            dst_des,
            self.feature_matcher_,
        )

        # Check transform
        valid_M, abs_det_M = utils.validate_warp_transform(
            M, self.homography_det_min)

        # Exit early if the warp didn't work
        if not valid_M:
            # Return more information on crash
            if self.debug:
                log = {
                    log_key: locals()[log_key]
                    for log_key in self.log_keys
                    if log_key in locals()
                }
            return 1, results, log

        # Warp the source image
        warped_img = cv2.warpPerspective(src_img, M, (x_size, y_size))

        # Combine the images
        blended_img = utils.blend_images(
            src_img=warped_img,
            dst_img=dst_img,
            fill_value=self.fill_value,
            outline=self.outline,
        )

        # Store the image
        self.save_image(blended_img, x_off, y_off)

        # Auxiliary: Convert to the dataset frame
        src_pts = cv2.KeyPoint_convert(src_kp)
        dsframe_src_pts = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2),
            M,
        ).reshape(-1, 2)
        dsframe_src_pts += np.array([x_off, y_off])
        results['dsframe_src_pts'] = dsframe_src_pts

        # Auxiliary: Convert bounding box (needed for georeferencing)
        (
            results['x_off'], results['y_off'],
            results['x_size'], results['y_size']
        ) = utils.warp_bounds(src_img, M)
        results['x_off'] += x_off
        results['y_off'] += y_off

        if self.debug:
            log = {
                log_key: locals()[log_key]
                for log_key in self.log_keys
                if log_key in locals()
            }
        return 0, results, log

    def close(self):

        self.reffed_mosaic.close()
        super().close()
