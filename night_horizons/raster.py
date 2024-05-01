'''Convenient handling of raster data (images, GDAL datasets).
This is largely not used in production, in favor of simpler, more-direct
calculations, but is very useful for testing and inspecting.

Because these are not production critical, documentation and testing are light.
'''
from typing import Union, Tuple

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
    '''Simple image class that can be used to store and manipulate images.
    '''

    io = data_io.ImageIO()

    def __init__(self, img: np.ndarray):
        '''
        Initialize the Image object.

        Parameters
        ----------
        img : np.ndarray
            The input image as a NumPy array.
        '''
        if np.issubdtype(img.dtype, np.floating):
            self.img = img.astype('float32')
        elif np.issubdtype(img.dtype, np.integer):
            max_val = np.iinfo(img.dtype).max
            # Convert to uint8, the expected type
            self.img_int = (img / max_val * 255).astype(np.uint8)
            self.img = (img / max_val).astype(np.float32)

    @classmethod
    def open(
        cls,
        fp: str,
        dtype: str = 'uint8',
        img_shape: Tuple[int] = (1200, 1920)
    ) -> "Image":
        '''
        Opens an image file and returns an Image object.

        Parameters
        ----------
        fp : str
            The file path of the image file to be opened.
        dtype : str, optional
            The data type of the image, defaults to 'uint8'.
        img_shape : Tuple[int], optional
            The shape of the image, defaults to (1200, 1920).

        Returns
        -------
        Image
            An Image object representing the opened image file.
        '''

        img = cls.io.load(fp, dtype=dtype, img_shape=img_shape)

        return Image(img)

    def save(self, fp: str, img: str = 'img_int'):
            '''
            Save the image array to a file.

            Parameters
            ----------
            fp : str
                The file path where the image will be saved.
            img : str, optional
                The name of the image array attribute to be saved.
                Default is 'img_int'.
            '''

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
        '''Integer format for the image.'''
        if not hasattr(self, '_img_int'):
            self._img_int = self.get_img_int_from_img()
        return self._img_int

    @img_int.setter
    def img_int(self, value):
        self._img_int = value

    @property
    def img_shape(self):
        '''Dimensions of the image.'''
        return self._img.shape[:2]

    @property
    def semitransparent_img(self) -> np.ndarray[float]:
        '''Version of the image with an alpha channel,
        where zero values are transparent.

        Returns
        -------
        np.ndarray[float]
            The image with an alpha channel.
        '''
        if not hasattr(self, '_semitransparent_img'):
            self._semitransparent_img = self.get_semitransparent_img()
        return self._semitransparent_img

    @property
    def semitransparent_img_int(self) -> np.ndarray[int]:
        '''Version of the image with an alpha channel,
        where zero values are transparent. This is the integer version.

        Returns
        -------
        np.ndarray[int]
            The image with an alpha channel.
        '''
        if not hasattr(self, '_semitransparent_img_int'):
            self._semitransparent_img_int = self.get_semitransparent_img_int()
        return self._semitransparent_img_int

    @property
    def kp(self):
        '''Keypoints for the image, used for feature matching.'''
        if not hasattr(self, '_kp'):
            self.get_features()
        return self._kp

    @property
    def des(self):
        '''Descriptors for the image, used for feature matching.'''
        if not hasattr(self, '_des'):
            self.get_features()
        return self._des

    def get_img_int_from_img(self) -> np.ndarray[int]:
        '''Convert an image to integer format.

        Returns
        -------
        np.ndarray[int]
            The image as an integer array.
        '''

        img_int = (self.img * 255).astype(np.uint8)

        return img_int

    def get_nonzero_mask(self) -> np.ndarray[bool]:
        '''Get a mask of the image where the values are nonzero.

        Returns
        -------
        np.ndarray[bool]
            A boolean array representing the mask where True indicates
            nonzero values in the image.
        '''

        return self.img_int.sum(axis=2) > 0

    def get_semitransparent_img(self) -> np.ndarray[float]:
        '''
        Returns a semitransparent image with an alpha channel.

        Returns
        -------
        np.ndarray[float]
            The semitransparent image with shape (height, width, 4),
            where the first three channels represent the RGB values
            of the original image, and the fourth channel represents
            the alpha (transparency) values.
        '''

        if self.img.shape[2] == 4:
            return self.img

        semitransparent_img = np.zeros(
            shape=(self.img_shape[0], self.img_shape[1], 4)
        )
        semitransparent_img[:, :, :3] = self.img
        semitransparent_img[:, :, 3] = self.get_nonzero_mask().astype(float)

        return semitransparent_img

    def get_semitransparent_img_int(self) -> np.ndarray[int]:
        '''
        Returns a semitransparent image with an alpha channel.

        Returns
        -------
        np.ndarray[int]
            The semitransparent image with shape (height, width, 4),
            where the first three channels represent the RGB values
            of the original image, and the fourth channel represents
            the alpha (transparency) values.
        '''

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

    def get_features(self) -> Tuple:
        '''Get keypoints and descriptors for the image.
        '''

        orb = cv2.ORB_create()

        self._kp, self._des = orb.detectAndCompute(self.img_int, None)

        return self._kp, self._des

    def get_pixel_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Get pixel coordinates for the image (ranging from 0 to width/height).
        '''

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
        '''Plot keypoints.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the keypoints. If not provided, a new figure and axes will be created.
        kp : list of cv2.KeyPoint, optional
            The keypoints to plot. If not provided, the keypoints stored in the object will be used.
        colors : array-like, optional
            The colors to use for each keypoint. If not provided, the response values of the keypoints will be used.
        crs_transform : callable, optional
            A coordinate transformation function that takes x and y coordinates as input and returns transformed coordinates.
        cmap : str or colormap, optional
            The colormap to use for coloring the keypoints. Default is 'viridis'.
        vmin : float, optional
            The minimum value for the colormap. If not provided, the minimum value of the colors will be used.
        vmax : float, optional
            The maximum value for the colormap. If not provided, the maximum value of the colors will be used.
        *args, **kwargs : additional arguments
            Additional arguments to be passed to the scatter plot function.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object representing the keypoints.

        '''

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
        Display the image on the specified axes.

        NOTE: This will not be consistent with imshow, because with imshow
        the y-axis increases downwards, consistent with old image
        processing schemes. Instead this is consistent with transposing and
        positively scaling the image to cartesian coordinates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to display the image. If not provided, a new figure and axes will be created.
        img : str, optional
            The name of the image attribute to display. Default is 'img'.
        img_transformer : object, optional
            An image transformer object that applies transformations to the image before displaying.
        downsample : float, optional
            The downsample factor for the figure size. Default is 240.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to be passed to the `pcolormesh` function.
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
    '''Image with georeferencing.'''

    io = data_io.ReferencedImageIO()
    dataset_io = data_io.GDALDatasetIO()

    def __init__(
        self,
        img,
        x_bounds,
        y_bounds,
        cart_crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        latlon_crs: Union[str, pyproj.CRS] = 'EPSG:4326',
    ):
        '''
        Initialize a Raster object.

        Parameters
        ----------
        img : ndarray
            The image data.
        x_bounds : tuple
            The bounds of the x-axis.
        y_bounds : tuple
            The bounds of the y-axis.
        cart_crs : str or pyproj.CRS, optional
            The coordinate reference system (CRS) for the Cartesian coordinates. Defaults to 'EPSG:3857'.
        latlon_crs : str or pyproj.CRS, optional
            The CRS for the latitude and longitude coordinates. Defaults to 'EPSG:4326'.
        '''

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
        '''
        Opens an image file and returns a ReferencedImage object.

        Parameters
        ----------
        fp : str
            The file path of the image file to be opened.
        cart_crs : Union[str, pyproj.CRS], optional
            The coordinate reference system (CRS) to be used for the Cartesian coordinates of the image. 
            It can be specified as a string (e.g., 'EPSG:3857') or as a pyproj.CRS object. 
            The default value is 'EPSG:3857'.
        latlon_crs : Union[str, pyproj.CRS], optional
            The coordinate reference system (CRS) to be used for the latitude and longitude coordinates of the image. 
            It can be specified as a string (e.g., 'EPSG:4326') or as a pyproj.CRS object. 
            The default value is 'EPSG:4326'.

        Returns
        -------
        ReferencedImage
            A ReferencedImage object representing the opened image, with associated coordinate reference systems.

        '''

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
        '''
        Save the raster image to a file.

        Parameters
        ----------
        fp : str
            The file path where the image will be saved.
        img_key : str, optional
            The key of the image array to be saved. Default is 'img_int'.
        '''

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
        '''Bounds of the image in latitude and longitude coordinates.'''
        if not hasattr(self, '_latlon_bounds'):
            self._latlon_bounds = self.get_bounds(self.latlon_crs)
        return self._latlon_bounds

    @property
    def cart_bounds(self):
        '''Bounds of the image in Cartesian coordinates.'''
        if not hasattr(self, '_cart_bounds'):
            self._cart_bounds = self.get_bounds(self.cart_crs)
        return self._cart_bounds

    @property
    def img_shape(self):
        '''Shape of the image.'''
        if hasattr(self, '_img'):
            return self._img.shape[:2]
        else:
            return (self.dataset.RasterYSize, self.dataset.RasterXSize)

    def set_projections(self, cart_crs, latlon_crs):
        '''Set the coordinate reference systems and transformations.'''

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
        '''Get the bounds of the image in the specified CRS.'''

        (
            x_bounds, y_bounds, pixel_width, pixel_height
        ) = self.dataset_io.get_bounds_from_dataset(self.dataset)

        return x_bounds, y_bounds

    def get_cart_coordinates(self):
        '''Get the Cartesian coordinates of the pixels.'''

        x_bounds, y_bounds = self.cart_bounds

        xs = np.linspace(x_bounds[0], x_bounds[1], self.img_shape[1])
        ys = np.linspace(y_bounds[1], y_bounds[0], self.img_shape[0])

        return xs, ys

    def get_pixel_widths(self):
        '''Get the pixel widths in the x and y directions.'''
        xs, ys = self.get_cart_coordinates()
        return np.abs(xs[1] - xs[0]), np.abs(ys[1] - ys[0])

    def get_pixel_coordinates(self):
        '''Get the coordinates of the pixels in pixel space.'''

        pxs = np.arange(self.dataset.RasterXSize)
        pys = np.arange(self.dataset.RasterYSize)

        return pxs, pys

    def convert_pixel_to_cart(self, pxs, pys):
        '''Convert the pixels to Cartesian coordinates.

        This method takes the pixel coordinates (pxs, pys) and converts them to Cartesian coordinates (xs, ys) based on the defined cart_bounds.

        Parameters
        ----------
        pxs : float or array-like
            The x-coordinates of the pixels.
        pys : float or array-like
            The y-coordinates of the pixels.

        Returns
        -------
        xs : float or array-like
            The x-coordinates in Cartesian coordinates.
        ys : float or array-like
            The y-coordinates in Cartesian coordinates.
        '''

        (x_min, x_max), (y_min, y_max) = self.cart_bounds

        x_scaling = (x_max - x_min) / (self.dataset.RasterXSize - 1)
        y_scaling = (y_min - y_max) / (self.dataset.RasterYSize - 1)

        xs = x_scaling * pxs + x_min
        ys = y_scaling * pys + y_max

        return xs, ys

    def convert_cart_to_pixel(self, xs, ys):
        '''Convert the Cartesian coordinates to pixels.

        Parameters
        ----------
        xs : float or array-like
            The x-coordinate(s) in Cartesian coordinates.
        ys : float or array-like
            The y-coordinate(s) in Cartesian coordinates.

        Returns
        -------
        pxs : float or array-like
            The x-coordinate(s) in pixel coordinates.
        pys : float or array-like
            The y-coordinate(s) in pixel coordinates.
        '''

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
        '''Plot the extent of the image.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the bounds.

        set_limits : bool, optional
            Whether to set the limits of the axes based on the bounds. Default is False.

        limits_padding : float, optional
            The padding factor to apply when setting the limits of the axes. Default is 0.1.

        *args, **kwargs
            Additional arguments and keyword arguments to pass to the `Rectangle` patch.
        '''

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
        '''
        Plot the keypoints on the image.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the Kp index. If not provided, a new figure and axes will be created.
        crs_transform : str or callable, optional
            The coordinate reference system (CRS) transformation to use. If 'cartesian', the default Cartesian transformation
            will be used. If a callable is provided, it should be a function that takes in the x and y coordinates and returns
            the transformed coordinates.
        *args : positional arguments
            Additional positional arguments to be passed to the underlying `super().plot_kp()` method.
        **kwargs : keyword arguments
            Additional keyword arguments to be passed to the underlying `super().plot_kp()` method.

        Returns
        -------
        The result of the underlying `super().plot_kp()` method.
        '''

        if crs_transform == 'cartesian':
            crs_transform = self.convert_pixel_to_cart

        return super().plot_kp(
            ax=ax,
            crs_transform=crs_transform,
            *args,
            **kwargs
        )

    def show(self, ax=None, img='img', crs='pixel', *args, **kwargs):
        '''Plot the image itself.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the image. If not provided, a new figure and axes will be created.
        img : str, optional
            The name of the image attribute to plot. Defaults to 'img'.
        crs : str, optional
            The coordinate reference system to use for plotting. Defaults to 'pixel'.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to the `pcolormesh` function.

        Returns
        -------
        None
            This method does not return anything.
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
        '''Add the raster image to a folium map.

        Parameters
        ----------
        m : folium map
            The map to add the raster image to.
        img : str, optional
            The name of the image attribute to use, by default 'semitransparent_img'
        label : str, optional
            The label/name of the raster image, by default 'referenced'
        include_corner_markers : bool, optional
            Whether to include corner markers on the map, by default False

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
