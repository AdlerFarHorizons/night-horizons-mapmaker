'''Convenient handling of raster data (images, GDAL datasets).
'''
from typing import Union

import cv2
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
import pyproj

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from . import data_io
from .transformers.raster import RasterCoordinateTransformer


class Image:

    io = data_io.ImageIO()

    def __init__(self, img):
        if np.issubdtype(img.dtype, np.floating):
            self.img = img.astype('float32')
        elif np.issubdtype(img.dtype, np.integer):
            max_val = np.iinfo(img.dtype).max
            # Convert to uint8, the expected type
            self.img_int = (img / max_val * 255).astype(np.uint8)
            self.img = (img / max_val).astype(np.float32)

    @classmethod
    def open(cls, fp, dtype: str = 'uint8', img_shape=(1200, 1920)):

        img = cls.io.load(fp, dtype=dtype, img_shape=img_shape)

        return Image(img)

    def save(self, fp, img='img_int'):

        img_arr = getattr(self, img)

        self.io.save(fp, img_arr)

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

    def show(
        self,
        ax=None,
        img='img',
        img_transformer=None,
        downsample=240.,
        *args,
        **kwargs
    ):
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
            fig = plt.figure(figsize=np.array(self.img_shape) / downsample)
            ax = plt.gca()

        pxs, pys = self.get_pixel_coordinates()

        img = getattr(self, img)
        if img_transformer is not None:
            img = img_transformer.fit_transform([img, ])[0]

        ax.pcolormesh(
            pxs,
            pys,
            img,
            *args,
            **kwargs
        )

        ax.set_aspect('equal')
        ax.set_xlim(pxs[0], pxs[-1])
        ax.set_ylim(pys[-1], pxs[0])


class ReferencedImage(Image):

    io = data_io.RegisteredImageIO()
    dataset_io = data_io.GDALDatasetIO()

    def __init__(
        self,
        img,
        x_bounds,
        y_bounds,
        cart_crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        latlon_crs: Union[str, pyproj.CRS] = 'EPSG:4326',
    ):

        cart_crs

        super().__init__(img)

        self.dataset = self.io.save(
            filepath='',
            img=self.img_int,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            crs=pyproj.CRS(cart_crs),
            driver='MEM',
            options=[],
        )

        # Set CRS properties
        self.set_projections(cart_crs, latlon_crs)

    @classmethod
    def open(
        cls,
        fp,
        cart_crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        latlon_crs: Union[str, pyproj.CRS] = 'EPSG:4326',
    ):

        cart_crs = pyproj.CRS(cart_crs)
        img, x_bounds, y_bounds = cls.io.load(fp, crs=cart_crs)

        return ReferencedImage(
            img,
            x_bounds,
            y_bounds,
            cart_crs=cart_crs,
            latlon_crs=latlon_crs,
        )

    def save(self, fp, img_key='img_int'):

        save_arr = getattr(self, img_key)
        x_bounds, y_bounds = self.cart_bounds

        self.io.save(
            filepath=fp,
            img=save_arr,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            crs=self.cart_crs,
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

    def set_projections(self, cart_crs, latlon_crs):

        # Establish CRS and conversions
        self.cart_crs = pyproj.CRS(cart_crs)
        self.latlon_crs = pyproj.CRS(latlon_crs)
        self.cart_to_latlon = pyproj.Transformer.from_crs(
            self.cart_crs,
            self.latlon_crs
        )
        self.latlon_to_cart = pyproj.Transformer.from_crs(
            self.latlon_crs,
            self.cart_crs
        )

    def get_bounds(self, crs: pyproj.CRS):

        (
            x_bounds, y_bounds, pixel_width, pixel_height
        ) = self.dataset_io.get_bounds_from_dataset(self.dataset)

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

    def plot_bounds(
        self,
        ax,
        set_limits=False,
        limits_padding=0.1,
        *args, **kwargs
    ):

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

        if set_limits:
            ax.set_xlim(
                x_bounds[0] - width * limits_padding,
                x_bounds[1] + width * limits_padding,
            )
            ax.set_ylim(
                y_bounds[0] - height * limits_padding,
                y_bounds[1] + height * limits_padding,
            )

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
                        ),
                    ).add_to(bounds_group)
            bounds_group.add_to(m)


class DatasetWrapper:
    '''Somewhat defunct wrapper for GDAL Dataset that better handles
    working with bounds and pixel resolutions instead of
    corners, array dimensions, and pixel resolutions.
    This is useful for mosaics, where the overlap really matters.
    '''

    io = data_io.GDALDatasetIO()

    def __init__(
        self,
        dataset: Union[str, gdal.Dataset],
        x_bounds: np.ndarray,
        y_bounds: np.ndarray,
        pixel_width: float,
        pixel_height: float,
        n_bands: int = 4,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        assert False, 'Deprecated'

        # Get dimensions
        width = x_bounds[1] - x_bounds[0]
        xsize = int(np.round(width / pixel_width))
        height = y_bounds[1] - y_bounds[0]
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

        # Set up the coordinate transformer
        self.transformer = RasterCoordinateTransformer()
        self.transformer.fit(
            X=None,
            y=None,
            dataset=self.dataset,
            crs=self.crs,
        )

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

        # Get bounds
        (
            x_bounds, y_bounds,
            pixel_width, pixel_height,
        ) = cls.io.get_bounds_from_dataset(
            dataset,
        )

        return cls(
            dataset,
            x_bounds, y_bounds,
            pixel_width, pixel_height,
            n_bands=dataset.RasterCount,
            crs=crs,
        )

    def get_image(self, x_min, x_max, y_min, y_max, trim: bool = False):
        
        x_off, y_off, x_size, y_size = self.transformer.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )
        x_off, y_off, x_size, y_size = self.transformer.handle_out_of_bounds(
            x_off, y_off, x_size, y_size, trim=trim,
        )

        img = self.dataset.ReadAsArray(
            xoff=x_off,
            yoff=y_off,
            xsize=x_size,
            ysize=y_size
        )
        return img.transpose(1, 2, 0)

    def save_image(self, img, x_min, x_max, y_min, y_max):

        x_off, y_off, x_size, y_size = self.transformer.physical_to_pixel(
            x_min, x_max, y_min, y_max
        )

        img_to_save = img.transpose(2, 0, 1)
        self.dataset_.WriteArray(
            img_to_save,
            xoff=x_off,
            yoff=y_off,
        )
