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


class Mosaic(TransformerMixin, BaseEstimator):
    '''Assemble a mosaic from georeferenced images.

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
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        padding: float = 0.,
        passthrough: Union[bool, list[str]] = False,
        exist_ok: bool = False,
        outline: int = 0,
    ):
        self.filepath = filepath
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.fill_value = fill_value
        self.dtype = dtype
        self.n_bands = n_bands
        self.padding = padding
        self.passthrough = passthrough
        self.exist_ok = exist_ok
        self.outline = outline

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

        # We'll decide on the iteration order based on proximity to
        # the central coords
        self.central_coords_ = X[['x_center', 'y_center']].mean().values

        # Load the dataset if it already exists
        if os.path.isfile(self.filepath):
            self.dataset_ = gdal.Open(self.filepath, gdal.GA_Update)

            # Get the dataset bounds
            (
                (self.x_min_, self.x_max_),
                (self.y_min_, self.y_max_),
                self.pixel_width_, self.pixel_height_
            ) = raster.get_bounds_from_dataset(
                self.dataset_,
                self.crs,
            )

            return self

        # Check the input is good.
        X = utils.check_df_input(
            X,
            ['filepath'] + preprocess.GEOTRANSFORM_COLS,
            passthrough=self.passthrough
        )
        if not self.exist_ok:
            if os.path.isfile(self.filepath):
                raise FileExistsError('File already exists at destination.')

        # Convert CRS as needed
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Get bounds
        self.x_min_ = X['x_min'].min() - self.padding
        self.x_max_ = X['x_max'].max() + self.padding
        self.y_min_ = X['y_min'].min() - self.padding
        self.y_max_ = X['y_max'].max() + self.padding

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
        self.pixel_height_ = -height / ysize

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

    def score(self, X, y=None, tm_metric=cv2.TM_CCOEFF_NORMED):

        self.scores_ = []
        for i, fp in enumerate(tqdm.tqdm(X['filepath'])):

            row = X.iloc[i]

            actual_img = utils.load_image(fp, dtype=self.dtype)
            mosaic_img = self.get_image(
                row['x_min'],
                row['x_max'],
                row['y_min'],
                row['y_max'],
            )

            r = metrics.image_to_image_ccoeff(
                actual_img,
                mosaic_img[:, :, :3],
                tm_metric=tm_metric,
            )
            self.scores_.append(r)

        score = np.median(self.scores_)

        return score

    def calc_iteration_indices(self, X):

        d_to_center = np.sqrt(
            (X['x_center'] - self.central_coords_[0])**2.
            + (X['y_center'] - self.central_coords_[1])**2.
        )
        iteration_indices = d_to_center.sort_values().index

        return iteration_indices

    def bounds_to_offset(self, x_min, x_max, y_min, y_max):

        # Get offsets
        x_offset = x_min - self.x_min_
        x_offset_count = int(np.round(x_offset / self.pixel_width_))
        y_offset = y_max - self.y_max_
        y_offset_count = int(np.round(y_offset / self.pixel_height_))

        # Get width counts
        xsize = int(np.round((x_max - x_min) / self.pixel_width_))
        ysize = int(np.round((y_max - y_min) / -self.pixel_height_))

        return x_offset_count, y_offset_count, xsize, ysize

    def get_image(self, x_min, x_max, y_min, y_max):

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

    def save_image(self, img, x_min, x_max, y_min, y_max):

        x_offset_count, y_offset_count, xsize, ysize = self.bounds_to_offset(
            x_min, x_max, y_min, y_max
        )

        img_to_save = img.transpose(2, 0, 1)
        self.dataset_.WriteArray(
            img_to_save,
            xoff=x_offset_count,
            yoff=y_offset_count
        )

    def blend_images(
        self,
        src_img,
        dst_img,
    ):

        # Fill value defaults to values that would be opaque
        fill_value = self.fill_value
        if fill_value is None:
            if np.issubdtype(dst_img.dtype, np.integer):
                fill_value = 255
            else:
                fill_value = 1.

        # Doesn't consider zeros in the final channel as empty
        n_bands = dst_img.shape[-1]
        is_empty = (dst_img[:, :, :n_bands - 1].sum(axis=2) == 0)

        # Blend
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

        # Add an outline
        if self.outline > 0:
            blended_img[:self.outline] = fill_value
            blended_img[-1 - self.outline:] = fill_value
            blended_img[:, :self.outline] = fill_value
            blended_img[:, -1 - self.outline:] = fill_value

        return blended_img

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

    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        X = utils.check_df_input(
            X,
            ['filepath'] + preprocess.GEOTRANSFORM_COLS,
            passthrough=self.passthrough
        )

        # Check if fit had been called
        check_is_fitted(self, 'dataset_')

        # Loop through and include
        for i, fp in enumerate(tqdm.tqdm(X['filepath'])):

            row = X.iloc[i]

            # Get data
            src_img = utils.load_image(
                fp,
                dtype=self.dtype,
            )
            dst_img = self.get_image(
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
            )

            # Store the image
            self.save_image(
                blended_img,
                row['x_min'],
                row['x_max'],
                row['y_min'],
                row['y_max'],
            )

        # Finish by flushing the cache
        self.dataset_.FlushCache()

        return self.dataset_


class LessReferencedMosaic(Mosaic):

    def __init__(
        self,
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: type = np.uint8,
        n_bands: int = 4,
        padding: float = 0.,
        passthrough: Union[bool, list[str]] = False,
        exist_ok: bool = True,
        outline: int = 0,
        homography_det_min=0.6,
    ):

        super().__init__(
            filepath=filepath,
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            padding=padding,
            passthrough=passthrough,
            exist_ok=exist_ok,
            outline=outline,
        )

        self.homography_det_min = homography_det_min

        self.feature_detector = cv2.ORB_create()
        self.feature_matcher = cv2.BFMatcher()

    def predict(
        self,
        X: pd.DataFrame,
        y=None,
        iteration_indices: np.ndarray[int] = None,
    ):
        # Useful for debugging
        self.log = {}

        X = utils.check_df_input(
            X,
            ['filepath'] + preprocess.GEOTRANSFORM_COLS,
            passthrough=self.passthrough
        )

        if iteration_indices is None:
            iteration_indices = self.calc_iteration_indices(X)

        # DEBUG
        # import pdb; pdb.set_trace()

        # Check if fit had been called
        check_is_fitted(self, 'dataset_')

        # Loop through and include
        self.log['return_codes'] = []
        for ind in tqdm.tqdm(iteration_indices):

            row = X.loc[ind]
            return_code, info = self.incorporate_image(row)
            self.log['return_codes'].append(return_code)

        # Finish by flushing the cache
        # TODO: This affects a fitted property, which is bad form.
        self.dataset_.FlushCache()

        return self.dataset_

    def incorporate_image(self, row, dsframe_dst_pts, dsframe_dst_des):

        # Get image location
        x_min = row['x_min'] - self.padding
        x_max = row['x_max'] + self.padding
        y_min = row['y_min'] - self.padding
        y_max = row['y_max'] + self.padding
        x_off, y_off, x_size, y_size = self.bounds_to_offset(
            x_min, x_max, y_min, y_max)

        # Get dst features
        in_bounds = self.check_bounds(
            dsframe_dst_pts,
            x_off, y_off, x_size, y_size
        )
        assert in_bounds.sum() > 0, \
            f'No image data in the search zone for index {row.name}'
        dst_pts = dsframe_dst_pts[in_bounds] - np.array([x_off, y_off])
        dst_kp = cv2.KeyPoint_convert(dst_pts)
        dst_des = dsframe_dst_des[in_bounds]

        # Get src features
        src_img = utils.load_image(
            row['filepath'],
            dtype=self.dtype,
        )
        src_kp, src_des = self.feature_detector.detectAndCompute(src_img, None)

        # Feature matching
        M, info = utils.calc_warp_transform(
            src_kp,
            src_des,
            dst_kp,
            dst_des,
            self.feature_matcher,
        )

        # Convert to the dataset frame
        src_pts = cv2.KeyPoint_convert(src_kp)
        dsframe_src_pts = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2),
            M,
        ).reshape(-1, 2)
        dsframe_src_pts += np.array([x_off, y_off])

        # Store
        info['dsframe_src_pts'] = dsframe_src_pts
        info['src_des'] = src_des

        # Exit early if the warp didn't work
        if not utils.validate_warp_transform(M, self.homography_det_min):
            # Return more information on crash
            info['M'] = M
            return 1, info

        # Warp the source image
        warped_img = cv2.warpPerspective(src_img, M, (x_size, y_size))

        # Combine the images
        dst_img = self.get_image(x_min, x_max, y_min, y_max)
        blended_img = self.blend_images(
            src_img=warped_img,
            dst_img=dst_img,
        )

        # Store the image
        self.save_image(blended_img, x_min, x_max, y_min, y_max)

        return 0, info
