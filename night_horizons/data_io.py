from abc import ABC, abstractmethod
import inspect
import os
import pickle
from typing import Tuple, Union

import cv2
import numpy as np
import osgeo
from osgeo import gdal, gdal_array
from osgeo.osr import SpatialReference
gdal.UseExceptions()
import pandas as pd
import pyproj
from pyproj.enums import WktVersion
import yaml


class DataIO(ABC):
    @abstractmethod
    def save(filepath, data):
        pass

    @abstractmethod
    def load(filepath):
        pass


class ImageIO(DataIO):
    name = 'image'

    @staticmethod
    def save(filepath, data):

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = data[:, :, ::-1]
        cv2.imwrite(filepath, data)

    @staticmethod
    def load(
        filepath: str,
        dtype: Union[str, type] = 'uint8',
        img_shape: Tuple = (1200, 1920),
    ) -> np.ndarray:
        '''Load an image from disk.

        Parameters
        ----------
            filepath
                Location of the image.
            dtype
                Datatype. Defaults to integer from 0 to 255.
            img_shape
                Image dimensions, used if loading a file of the '.raw' type.
        Returns
        -------
            img
                Image as a numpy array.
        '''

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'File {filepath} not found')

        ext = os.path.splitext(filepath)[1]

        # Load and reshape raw image data.
        if ext == '.raw':

            raw_img = np.fromfile(filepath, dtype=np.uint16)
            raw_img = raw_img.reshape(img_shape)

            img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)
            img_max = 2**12 - 1

        elif ext in ['.tiff', '.tif']:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            # CV2 defaults to BGR, but RGB is more standard for our purposes
            img = img[:, :, ::-1]
            img_max = np.iinfo(img.dtype).max

        else:
            raise IOError('Cannot read filetype {}'.format(ext))

        # TODO: Delete this
        # if img is None:
        #     return img

        if isinstance(dtype, str):
            dtype = getattr(np, dtype)

        # Check if conversion needs to be done
        if img.dtype == dtype:
            return img

        # Rescale
        img = img / img_max
        img = (img * np.iinfo(dtype).max).astype(dtype)

        return img


class GDALDatasetIO(DataIO):
    name = 'gdal_dataset'

    @staticmethod
    def save(filepath, data, driver: str = 'GTiff'):
        '''The alternative is to flush the cache and then re-open from disk.

        Parameters
        ----------
        Returns
        -------
        '''

        # Create a copy with a driver that saves to disk
        driver = gdal.GetDriverByName('GTiff')
        save_dataset = driver.CreateCopy(filepath, data, 0)
        save_dataset.FlushCache()
        save_dataset = None

    @staticmethod
    def load(
        filepath: str,
        mode: int = gdal.GA_ReadOnly,
        crs: pyproj.CRS = None
    ) -> gdal.Dataset:
        data = gdal.Open(filepath, mode)

        # Convert to desired crs.
        if crs is not None:
            data = GDALDatasetIO.convert(data, crs)

        return data

    @staticmethod
    def load_from_viirs_hdf5(
        filepath: str,
        output_filepath: str = None,
    ) -> gdal.Dataset:
        '''Load a VIIRS HDF5 file.

        Parameters
        ----------
        filepath
            Path to the VIIRS HDF5 file.
        crs
            Desired coordinate system. Defaults to None, which means the
            coordinate system of the VIIRS file will be used.

        Returns
        -------
        data
            Image data.
        '''

        if output_filepath is None:
            output_filepath = filepath.replace('.h5', '.tiff')

        # If the conversion is already done
        if os.path.isfile(output_filepath):
            return GDALDatasetIO.load(output_filepath)

        # Open HDF file
        hdflayer = gdal.Open(filepath, gdal.GA_ReadOnly)

        # Open raster layer
        # hdflayer.GetSubDatasets()[0][0] - for first layer
        # hdflayer.GetSubDatasets()[1][0] - for second layer ...etc
        subhdflayer = hdflayer.GetSubDatasets()[0][0]
        rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

        # collect bounding box coordinates
        HorizontalTileNumber = int(
            rlayer.GetMetadata_Dict()["HorizontalTileNumber"]
        )
        VerticalTileNumber = int(
            rlayer.GetMetadata_Dict()["VerticalTileNumber"]
        )
            
        WestBoundCoord = (10 * HorizontalTileNumber) - 180
        NorthBoundCoord = 90 - (10 * VerticalTileNumber)
        EastBoundCoord = WestBoundCoord + 10
        SouthBoundCoord = NorthBoundCoord - 10

        # WGS84
        EPSG = "-a_srs EPSG:4326"

        translateOptionText = (
            EPSG
            + " -a_ullr "
            + str(WestBoundCoord)
            + " " + str(NorthBoundCoord)
            + " " + str(EastBoundCoord)
            + " " + str(SouthBoundCoord)
        )
        translateoptions = gdal.TranslateOptions(
            gdal.ParseCommandLine(translateOptionText)
        )

        translated = gdal.Translate(
            output_filepath,
            rlayer,
            options=translateoptions
        )

        return translated

    @staticmethod
    def create(
        filepath,
        x_min,
        y_max,
        pixel_width,
        pixel_height,
        crs,
        x_size,
        y_size,
        n_bands: int = 4,
        driver: str = 'MEM',
        return_dataset: bool = True,
        options: list[str] = ['TILED=YES'],
        *args, **kwargs
    ):

        # Initialize an empty GeoTiff
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            filepath,
            xsize=x_size,
            ysize=y_size,
            bands=n_bands,
            options=options,
            *args, **kwargs
        )

        # Properties
        dataset.SetProjection(crs.to_wkt())
        dataset.SetGeoTransform([
            x_min,
            pixel_width,
            0.,
            y_max,
            0.,
            pixel_height,
        ])
        if n_bands == 4:
            dataset.GetRasterBand(4).SetMetadataItem('Alpha', '1')

        if (driver == 'MEM') or return_dataset:
            return dataset

        # Close out the dataset for now. (Reduces likelihood of mem leaks.)
        dataset.FlushCache()
        dataset = None

    @staticmethod
    def convert(dataset: gdal.Dataset, target_crs: pyproj.CRS):

        # Check current CRS
        current_crs = pyproj.CRS(dataset.GetProjection())
        if current_crs == target_crs:
            return dataset

        # Change CRS to SRS so gdal knows what to do
        srs = SpatialReference()
        if osgeo.version_info.major < 3:
            srs.ImportFromWkt(target_crs.to_wkt(WktVersion.WKT1_GDAL))
        else:
            srs.ImportFromWkt(target_crs.to_wkt())

        converted_dataset = gdal.Warp(
            '',
            dataset,
            format='MEM',
            dstSRS=srs,
        )

        return converted_dataset

    @staticmethod
    def get_bounds_from_dataset(
        dataset: gdal.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, pyproj.CRS]:
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
        x_bounds = [x_min, x_max]
        y_bounds = [y_min, y_max]

        return (
            x_bounds,
            y_bounds,
            pixel_width,
            pixel_height,
        )


class RegisteredImageIO(DataIO):
    name = 'registered_image'

    @staticmethod
    def save(
        filepath,
        img,
        x_bounds,
        y_bounds,
        crs,
        driver='GTiff',
        *args, **kwargs
    ):

        if filepath != '':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Get data type
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(img.dtype)

        # Get pixel size
        pixel_width = (x_bounds[1] - x_bounds[0]) / img.shape[1]
        pixel_height = -(y_bounds[1] - y_bounds[0]) / img.shape[0]

        dataset = GDALDatasetIO.create(
            filepath=filepath,
            x_min=x_bounds[0],
            y_max=y_bounds[1],
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            crs=crs,
            x_size=img.shape[1],
            y_size=img.shape[0],
            n_bands=img.shape[2],
            driver=driver,
            eType=gdal_dtype,
            *args, **kwargs
        )

        # Write to the dataset
        dataset.WriteArray(img.transpose(2, 0, 1))

        # Stop and return if we want to keep the dataset open
        if driver == 'MEM':
            return dataset

        dataset.FlushCache()
        dataset = None

    @staticmethod
    def load(filepath, crs: pyproj.CRS = None):

        # Get image
        dataset = GDALDatasetIO.load(filepath, crs=crs)
        img = dataset.ReadAsArray()

        # For multiple bands, format accordingly
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)

        # Get bounds
        (
            x_bounds, y_bounds, dx, dy
        ) = GDALDatasetIO.get_bounds_from_dataset(dataset)

        data = (img, x_bounds, y_bounds)

        return data


class TabularIO(DataIO):
    name = 'tabular'

    @staticmethod
    def save(filepath, data):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath):
        df = pd.read_csv(filepath)
        return df


class YAMLIO(DataIO):
    name = 'yaml'

    @staticmethod
    def save(filepath, data):

        fullargspec = inspect.getfullargspec(type(data))
        settings = {}
        for setting in fullargspec.args:
            if setting == 'self':
                continue
            value = getattr(data, setting)
            try:
                pickle.dumps(value)
            except TypeError:
                value = 'no string repr'
            settings[setting] = value
        with open(filepath, 'w') as file:
            yaml.dump(settings, file)

    @staticmethod
    def load(filepath):
        pass
