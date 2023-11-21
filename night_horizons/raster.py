'''Convenient handling of raster data (images, GDAL datasets).
'''
import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

import copy
import glob
import os

import cv2
from osgeo import gdal, gdal_array
import pyproj

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from typing import Tuple, Union


class Image:

    def __init__(self, img):
        if np.issubdtype(img.dtype, np.floating):
            self.img = img
        elif np.issubdtype(img.dtype, np.integer):
            max_val = np.iinfo(img.dtype).max
            # Convert to uint8, the expected type
            self.img_int = (img / max_val * 255).astype(np.uint8)
            self.img = (img / max_val).astype(np.float32)

    @classmethod
    def open(cls, fp):

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        max_val = np.iinfo(img.dtype).max
        img = img[:, :, ::-1] / max_val  # Formatting

        return Image(img)

    @property
    def img(self):
        '''Image property for quick access. For the base class Image
        it's very simple, but will be overwritten by other classes.
        '''
        return self._img

    @img.setter
    def img(self, value):
        self._img = value

    @property
    def img_int(self) -> np.ndarray[int]:
        if not hasattr(self, '_img_int'):
            self._img_int = self.get_img_int_from_img()
        return self._img_int

    @img_int.setter
    def img_int(self, value):
        self._img_int = value

    @property
    def img_shape(self):
        return self._img.shape[:2]

    @property
    def semitransparent_img(self) -> np.ndarray[float]:
        if not hasattr(self, '_semitransparent_img'):
            self._semitransparent_img = self.get_semitransparent_img()
        return self._semitransparent_img

    @property
    def semitransparent_img_int(self) -> np.ndarray[int]:
        if not hasattr(self, '_semitransparent_img_int'):
            self._semitransparent_img_int = self.get_semitransparent_img_int()
        return self._semitransparent_img_int

    @property
    def kp(self):
        if not hasattr(self, '_kp'):
            self.get_features()
        return self._kp

    @property
    def des(self):
        if not hasattr(self, '_des'):
            self.get_features()
        return self._des

    def get_img_int_from_img(self) -> np.ndarray[int]:
        '''

        TODO: State (and assess) general principle--
            will use default options for image retrieval.
            For more fine-grained control call get_img
            first, instead of passing in additional arguments.
        '''

        img_int = (self.img * 255).astype(np.uint8)

        return img_int

    def get_nonzero_mask(self) -> np.ndarray[bool]:

        return self.img_int.sum(axis=2) > 0

    def get_semitransparent_img(self) -> np.ndarray[float]:

        if self.img.shape[2] == 4:
            return self.img

        semitransparent_img = np.zeros(
            shape=(self.img_shape[0], self.img_shape[1], 4)
        )
        semitransparent_img[:, :, :3] = self.img
        semitransparent_img[:, :, 3] = self.get_nonzero_mask().astype(float)

        return semitransparent_img

    def get_semitransparent_img_int(self) -> np.ndarray[int]:

        if self.img_int.shape[2] == 4:
            return self.img_int

        semitransparent_img_int = np.zeros(
            shape=(self.img_shape[0], self.img_shape[1], 4),
            dtype=np.uint8,
        )
        semitransparent_img_int[:, :, :3] = self.img_int
        semitransparent_img_int[:, :, 3] = (
            255 * self.get_nonzero_mask().astype(np.uint8)
        )

        return semitransparent_img_int

    def get_features(self):

        orb = cv2.ORB_create()

        self._kp, self._des = orb.detectAndCompute(self.img_int, None)

        return self._kp, self._des

    def get_pixel_coordinates(self):

        pxs = np.arange(self.img_shape[1])
        pys = np.arange(self.img_shape[0])

        return pxs, pys

    def plot_kp(
        self,
        ax=None,
        kp=None,
        colors=None,
        crs_transform=None,
        cmap='viridis',
        vmin=None,
        vmax=None,
        *args,
        **kwargs
    ):

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        # KP details retrieval
        if kp is None:
            kp = self.kp
        kp_xs, kp_ys = np.array([_.pt for _ in kp]).transpose()
        if colors is None:
            colors = np.array([_.response for _ in kp])

        # Transform to appropriate coordinate system
        if crs_transform is not None:
            kp_xs, kp_ys = crs_transform(kp_xs, kp_ys)

        # Colormap
        cmap = sns.color_palette(cmap, as_cmap=True)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Argument update
        used_kwargs = {
            'c': 'none',
            'marker': 'o',
            's': 150,
            'linewidth': 2,
        }
        used_kwargs.update(kwargs)

        # Plot itself
        s = ax.scatter(
            kp_xs,
            kp_ys,
            edgecolors=cmap(norm(colors)),
            *args,
            **used_kwargs
        )

        return s

    def show(self, ax=None, img='img', *args, **kwargs):
        '''
            NOTE: This will not be consistent with imshow, because with imshow
        the y-axis increases downwards, consistent with old image
        processing schemes. Instead this is consistent with transposing and
        positively scaling the image to cartesian coordinates.

        Args:
        Kwargs:
        Returns:
        '''

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        pxs, pys = self.get_pixel_coordinates()

        ax.pcolormesh(
            pxs,
            pys,
            getattr(self, img),
            *args,
            **kwargs
        )

        ax.set_aspect('equal')
        ax.set_xlim(pxs[0], pxs[-1])
        ax.set_ylim(pys[-1], pxs[0])


class ReferencedImage(Image):

    def __init__(
        self,
        img,
        x_bounds,
        y_bounds,
        cart_crs_code: str = 'EPSG:3857',
        latlon_crs_code: str = 'EPSG:4326',
    ):

        super().__init__(img)

        # Make a gdal dataset
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(
            self.img_int.dtype
        )
        driver = gdal.GetDriverByName('MEM')
        dataset = driver.Create(
            '',
            self.img_int.shape[1],
            self.img_int.shape[0],
            self.img_int.shape[2],
            gdal_dtype
        )

        self.dataset = dataset

        # Write data
        for i in range(self.img_int.shape[2]):
            self.dataset.GetRasterBand(i + 1).WriteArray(self.img_int[:, :, i])

        # Set CRS properties
        self.set_projections(cart_crs_code, latlon_crs_code)

        # Set geotransform
        dx = (x_bounds[1] - x_bounds[0]) / self.img_shape[1]
        dy = (y_bounds[1] - y_bounds[0]) / self.img_shape[0]
        geotransform = (
            x_bounds[0],
            dx,
            0,
            y_bounds[1],
            0,
            -dy
        )
        self.dataset.SetGeoTransform(geotransform)

    @classmethod
    def open(
        cls,
        filename,
        cart_crs_code: str = 'EPSG:3857',
        latlon_crs_code: str = 'EPSG:4326',
    ):

        # Get image
        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        img = dataset.ReadAsArray().transpose(1, 2, 0)

        # Get bounds
        cart_crs = pyproj.CRS(cart_crs_code)
        x_bounds, y_bounds, _, _ = get_bounds_from_dataset(dataset, cart_crs)

        return ReferencedImage(
            img,
            x_bounds,
            y_bounds,
            cart_crs_code=cart_crs_code,
            latlon_crs_code=latlon_crs_code
        )

    @property
    def latlon_bounds(self):
        if not hasattr(self, '_latlon_bounds'):
            self._latlon_bounds = self.get_bounds(self.latlon_crs)
        return self._latlon_bounds

    @property
    def cart_bounds(self):
        if not hasattr(self, '_cart_bounds'):
            self._cart_bounds = self.get_bounds(self.cart_crs)
        return self._cart_bounds

    @property
    def img_shape(self):
        if hasattr(self, '_img'):
            return self._img.shape[:2]
        else:
            return (self.dataset.RasterYSize, self.dataset.RasterXSize)

    def save(self, fp, img_key='img_int'):

        # Store to in-memory data first
        save_arr = getattr(self, img_key).transpose(2, 0, 1)
        self.dataset.WriteArray(save_arr)

        # Create a copy with a driver that saves to disk
        save_driver = gdal.GetDriverByName('GTiff')
        save_dataset = save_driver.CreateCopy(fp, self.dataset, 0)
        save_dataset.FlushCache()

    def set_projections(self, cart_crs_code, latlon_crs_code):

        # Establish CRS and conversions
        self.cart_crs_code = cart_crs_code
        self.latlon_crs_code = latlon_crs_code
        self.cart_crs = pyproj.CRS(cart_crs_code)
        self.latlon_crs = pyproj.CRS(latlon_crs_code)
        self.cart_to_latlon = pyproj.Transformer.from_crs(
            self.cart_crs,
            self.latlon_crs
        )
        self.latlon_to_cart = pyproj.Transformer.from_crs(
            self.latlon_crs,
            self.cart_crs
        )
        self.dataset.SetProjection(self.cart_crs.to_wkt())

    def get_bounds(self, crs: pyproj.CRS):

        x_bounds, y_bounds, _, _ = get_bounds_from_dataset(self.dataset, crs)

        return x_bounds, y_bounds

    def get_cart_coordinates(self):

        x_bounds, y_bounds = self.cart_bounds

        xs = np.linspace(x_bounds[0], x_bounds[1], self.img_shape[1])
        ys = np.linspace(y_bounds[1], y_bounds[0], self.img_shape[0])

        return xs, ys

    def get_pixel_widths(self):
        xs, ys = self.get_cart_coordinates()
        return np.abs(xs[1] - xs[0]), np.abs(ys[1] - ys[0])

    def get_pixel_coordinates(self):

        pxs = np.arange(self.dataset.RasterXSize)
        pys = np.arange(self.dataset.RasterYSize)

        return pxs, pys

    def convert_pixel_to_cart(self, pxs, pys):

        (x_min, x_max), (y_min, y_max) = self.cart_bounds

        x_scaling = (x_max - x_min) / (self.dataset.RasterXSize - 1)
        y_scaling = (y_min - y_max) / (self.dataset.RasterYSize - 1)

        xs = x_scaling * pxs + x_min
        ys = y_scaling * pys + y_max

        return xs, ys

    def convert_cart_to_pixel(self, xs, ys):

        (x_min, x_max), (y_min, y_max) = self.cart_bounds

        x_scaling = (self.dataset.RasterXSize - 1) / (x_max - x_min)
        y_scaling = (self.dataset.RasterYSize - 1) / (y_min - y_max)

        pxs = (xs - x_min) * x_scaling
        pys = (ys - y_max) * y_scaling

        return pxs, pys

    def plot_bounds(self, ax, *args, **kwargs):

        used_kwargs = {
            'linewidth': 3,
            'facecolor': 'none',
            'edgecolor': '#dd8452',
        }
        used_kwargs.update(kwargs)

        x_bounds, y_bounds = self.cart_bounds
        x_min = x_bounds[0]
        y_min = y_bounds[0]
        width = x_bounds[1] - x_bounds[0]
        height = y_bounds[1] - y_bounds[0]

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            *args,
            **used_kwargs
        )
        ax.add_patch(rect)

    def plot_kp(self, ax=None, crs_transform='cartesian', *args, **kwargs):

        if crs_transform == 'cartesian':
            crs_transform = self.convert_pixel_to_cart

        return super().plot_kp(
            ax=ax,
            crs_transform=crs_transform,
            *args,
            **kwargs
        )

    def show(self, ax=None, img='img', crs='pixel', *args, **kwargs):
        '''
        TODO: Make this more consistent with naming of other functions?
        '''

        # Use existing functionality
        if crs == 'pixel':
            return super().show(ax=ax, img=img, *args, **kwargs)

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img_shape) / 60.)
            ax = plt.gca()

        xs, ys = self.get_cart_coordinates()

        ax.pcolormesh(
            xs,
            ys,
            getattr(self, img),
            *args,
            **kwargs
        )

        ax.set_aspect('equal')

    def add_to_folium_map(
        self,
        m,
        img: str = 'semitransparent_img',
        label: str = 'referenced',
        include_corner_markers: bool = False
    ):
        '''Add to a folium map.
        
        Args:
            m (folium map): The map to add to.
        '''

        # Let's keep this as an optional import for now.
        import folium

        lon_bounds, lat_bounds = self.latlon_bounds
        bounds = [
            [lat_bounds[0], lon_bounds[0]],
            [lat_bounds[1], lon_bounds[1]]
        ]
        img_arr = getattr(self, img)

        folium.raster_layers.ImageOverlay(
            img_arr,
            bounds=bounds,
            name=label,
        ).add_to(m)

        # Markers for the corners so we can understand how the image pixels
        # get flipped around
        if include_corner_markers:
            bounds_group = folium.FeatureGroup(name=f'{label} bounds')
            minmax_labels = ['min', 'max']
            for ii in range(2):
                for jj in range(2):
                    hover_text = (
                        f'(x_{minmax_labels[jj]}, '
                        f'y_{minmax_labels[ii]})'
                    )
                    folium.Marker(
                        [lat_bounds[ii], lon_bounds[jj]],
                        popup=hover_text,
                        icon=folium.Icon(
                            color='gray',
                            # TODO: Incorporate colors
                            # icon_color=palette.as_hex()[jj * 2 + ii]
                        ),
                    ).add_to(bounds_group)
            bounds_group.add_to(m)


class BoundsDataset(gdal.Dataset):
    '''Wrapper for GDAL Dataset that better handles
    working with bounds and pixel resolutions instead of
    corners, array dimensions, and pixel resolutions.
    This is useful for mosaics, where the overlap really matters.
    '''

    def __init__(
        self,
        dataset: Union[str, gdal.Dataset],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        pixel_width: float,
        pixel_height: float,
        n_bands: int = 4,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        # Get dimensions
        width = x_max - x_min
        xsize = int(np.round(width / pixel_width))
        height = y_max - y_min
        ysize = int(np.round(height / -pixel_height))

        # Re-record pixel values to account for rounding
        pixel_width = width / xsize
        pixel_height = -height / ysize

        # Initialize an empty dataset
        if isinstance(dataset, str):
            driver = gdal.GetDriverByName('GTiff')
            self.dataset = driver.Create(
                dataset,
                xsize=xsize,
                ysize=ysize,
                bands=n_bands,
                options=['TILED=YES']
            )
        else:
            self.dataset = dataset

        # Properties
        if isinstance(crs, str):
            crs = pyproj.CRS(crs)
        self.SetProjection(crs.to_wkt())
        self.SetGeoTransform([
            x_min,
            pixel_width,
            0.,
            y_max,
            0.,
            pixel_height,
        ])
        if n_bands == 4:
            self.GetRasterBand(4).SetMetadataItem('Alpha', '1')

        # Store properties
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.n_bands = n_bands
        self.crs = crs

    @classmethod
    def open(
        cls,
        filename: str,
        crs: Union[str, pyproj.CRS] = None,
        *args,
        **kwargs
    ):

        # Open up the file
        dataset = gdal.Open(filename, *args, **kwargs)

        # CRS handling
        if isinstance(crs, str):
            crs = pyproj.CRS(crs)
        if crs is None:
            crs = pyproj.CRS(dataset.GetProjection())

        # Get bounds
        (
            x_min, x_max,
            y_min, y_max,
            pixel_width, pixel_height
        ) = get_bounds_from_dataset(
            dataset,
            crs,
        )

        return cls(
            dataset,
            x_min, x_max,
            y_min, y_max,
            pixel_width, pixel_height,
            n_bands=dataset.RasterCount,
            crs=crs,
        )

    def physical_to_pixel(self, x_min, x_max, y_min, y_max):

        # Get offsets
        x_offset = x_min - self.x_min_
        x_offset_count = int(np.round(x_offset / self.pixel_width_))
        y_offset = y_max - self.y_max_
        y_offset_count = int(np.round(y_offset / -self.pixel_height_))

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

        x_offset_count, y_offset_count, xsize, ysize = self.physical_to_pixel(
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

        x_offset_count, y_offset_count, xsize, ysize = self.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )

        img_to_save = img.transpose(2, 0, 1)
        self.dataset_.WriteArray(
            img_to_save,
            xoff=x_offset_count,
            yoff=y_offset_count
        )


# DEBUG
#    def __init__(
#        self,
#        filename: str,
#        x_bounds: Tuple[float, float],
#        y_bounds: Tuple[float, float],
#        pixel_width: float,
#        pixel_height: float,
#        crs: pyproj.CRS,
#        n_bands: int = 4,
#    ):
#
#        # Initialize an empty GeoTiff
#        xsize = int(np.round((x_bounds[1] - x_bounds[0]) / pixel_width))
#        ysize = int(np.round((y_bounds[1] - y_bounds[0]) / pixel_height))
#        driver = gdal.GetDriverByName('GTiff')
#        self.dataset = driver.Create(
#            filename,
#            xsize=xsize,
#            ysize=ysize,
#            bands=n_bands,
#            options=['TILED=YES']
#        )
#
#        # Properties
#        self.dataset.SetProjection(crs.to_wkt())
#        self.dataset.SetGeoTransform([
#            x_bounds[0],
#            pixel_width,
#            0.,
#            y_bounds[1],
#            0.,
#            -pixel_height,
#        ])
#        if n_bands == 4:
#            self.dataset.GetRasterBand(4).SetMetadataItem('Alpha', '1')
#
#        self.x_bounds = x_bounds
#        self.y_bounds = y_bounds
#        self.pixel_width = pixel_width
#        self.pixel_height = pixel_height
#        self.crs = crs
#        self.filename = filename
#        self.n_bands = n_bands
#
#    @classmethod
#    def open(cls, filename: str, crs: pyproj.CRS = None, *args, **kwargs):
#
#        dataset = cls.__new__(cls)
#        dataset.dataset = gdal.Open(filename, *args, **kwargs)
#
#        # CRS handling
#        if isinstance(crs, str):
#            crs = pyproj.CRS(crs)
#        if crs is None:
#            crs = pyproj.CRS(dataset.dataset.GetProjection())
#        else:
#            dataset.dataset.SetProjection(crs.to_wkt())
#        dataset.crs = crs
#        dataset.filename = filename
#
#        # Get bounds
#        (
#            dataset.x_bounds,
#            dataset.y_bounds,
#            dataset.pixel_width,
#            dataset.pixel_height
#        ) = get_bounds_from_dataset(
#            dataset.dataset,
#            crs,
#        )
#
#        return dataset
#
#    def physical_to_pixel(self, x_bounds, y_bounds):
#
#        # Get offsets
#        x_offset = x_bounds[0] - self.x_bounds[0]
#        x_offset_count = int(np.round(x_offset / self.pixel_width))
#        y_offset = self.y_bounds[1] - y_bounds[1]
#        y_offset_count = int(np.round(y_offset / self.pixel_height))
#
#        # Get width counts
#        xsize = int(np.round((x_bounds[1] - x_bounds[0]) / self.pixel_width))
#        ysize = int(np.round((y_bounds[1] - y_bounds[0]) / self.pixel_height))
#
#        return x_offset_count, y_offset_count, xsize, ysize
#
#    def get_img(self, x_bounds, y_bounds):
#
#        # Out of bounds
#        if (
#            (x_bounds[0] > self.x_bounds[1])
#            or (x_bounds[1] < self.x_bounds[0])
#            or (y_bounds[0] > self.y_bounds[1])
#            or (y_bounds[1] < self.y_bounds[0])
#        ):
#            raise ValueError(
#                'Tried to retrieve data fully out-of-bounds.'
#            )
#
#        # Only partially out-of-bounds
#        if x_bounds[0] < self.x_bounds[0]:
#            x_bounds[0] = self.x_bounds[0]
#        if x_bounds[1] > self.x_bounds[1]:
#            x_bounds[1] = self.x_bounds[1]
#        if y_bounds[0] < self.y_bounds[0]:
#            y_bounds[0] = self.y_bounds[0]
#        if y_bounds[1] > self.y_bounds[1]:
#            y_bounds[1] = self.y_bounds[1]
#
#        x_offset_count, y_offset_count, xsize, ysize = self.physical_to_pixel(
#            x_bounds,
#            y_bounds,
#        )
#
#        img = self.dataset.ReadAsArray(
#            xoff=x_offset_count,
#            yoff=y_offset_count,
#            xsize=xsize,
#            ysize=ysize
#        )
#        return img.transpose(1, 2, 0)
#
#    def get_referenced_image(self, x_bounds, y_bounds):
#
#        img = self.get_img(x_bounds, y_bounds)
#
#        reffed_image = ReferencedImage(
#            img,
#            x_bounds,
#            y_bounds,
#            cart_crs_code='{}:{}'.format(*self.crs.to_authority()),
#        )
#
#        return reffed_image
#
#    def flush_cache_and_close(self):
#
#        self.dataset.FlushCache()
#        self.dataset = None
#
#    def save_img(self, img, x_bounds, y_bounds):
#        '''
#        NOTE: You must call self.flush_cache_and_close to finish saving to disk.
#        '''
#
#        x_offset_count, y_offset_count, xsize, ysize = self.physical_to_pixel(
#            x_bounds,
#            y_bounds,
#        )
#
#        img_to_save = img.transpose(2, 0, 1)
#        self.dataset.WriteArray(img_to_save, xoff=x_offset_count, yoff=y_offset_count)


def get_bounds_from_dataset(
    dataset: gdal.Dataset,
    crs: pyproj.CRS
) -> Tuple[float]:
    '''Get image bounds in a given coordinate system.

    Args:
        crs: Desired coordinate system.

    Returns:
        x_bounds: x_min, x_max of the image in the target coordinate system
        y_bounds: y_min, y_max of the image in the target coordinate system
        pixel_width
        pixel_height
    '''

    # Get the coordinates
    x_min, pixel_width, x_rot, y_max, y_rot, pixel_height = \
        dataset.GetGeoTransform()

    # Get bounds
    x_max = x_min + pixel_width * dataset.RasterXSize
    y_min = y_max + pixel_height * dataset.RasterYSize

    # Convert to desired crs.
    dataset_crs = pyproj.CRS(dataset.GetProjection())
    dataset_to_desired = pyproj.Transformer.from_crs(
        dataset_crs,
        crs,
        always_xy=True
    )
    x_bounds, y_bounds = dataset_to_desired.transform(
        [x_min, x_max],
        [y_min, y_max],
    )
    pixel_width, pixel_height = dataset_to_desired.transform(
        pixel_width,
        pixel_height,
    )

    return (
        x_bounds,
        y_bounds,
        pixel_width, pixel_height
    )


def get_containing_bounds(reffed_images, crs, bordersize=0):

    # Pixel size
    all_x_bounds = []
    all_y_bounds = []
    pixel_widths = []
    pixel_heights = []
    for i, reffed_image_i in enumerate(reffed_images):

        # Bounds
        x_bounds_i, y_bounds_i = reffed_image_i.get_bounds(crs)
        all_x_bounds.append(x_bounds_i)
        all_y_bounds.append(y_bounds_i)

        # Pixel properties
        pixel_width, pixel_height = reffed_image_i.get_pixel_widths()
        pixel_widths.append(pixel_width)
        pixel_heights.append(pixel_height)

    # Containing bounds
    all_x_bounds = np.array(all_x_bounds)
    all_y_bounds = np.array(all_y_bounds)
    x_bounds = [all_x_bounds[:, 0].min(), all_x_bounds[:, 1].max()]
    y_bounds = [all_y_bounds[:, 0].min(), all_y_bounds[:, 1].max()]

    # Use median pixel properties
    pixel_width = np.median(pixel_widths)
    pixel_height = np.median(pixel_heights)

    if bordersize != 0:
        x_bounds[0] -= bordersize * pixel_width
        x_bounds[1] += bordersize * pixel_width
        y_bounds[0] -= bordersize * pixel_height
        y_bounds[1] += bordersize * pixel_height

    return x_bounds, y_bounds, pixel_width, pixel_height
