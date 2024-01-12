'''TODO: Finish implementing this.
'''
from abc import ABC, abstractmethod
import inspect
import os
import pickle
from typing import Tuple

import cv2
from osgeo import gdal, gdal_array
import pandas as pd
import pyproj
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

        data = data[:, :, ::-1]
        cv2.imwrite(filepath, data)

    @staticmethod
    def load(filepath):

        data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        data = data[:, :, ::-1]

        return data


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
    def load(filepath):
        data = gdal.Open(filepath, gdal.GA_ReadOnly)
        return data

    @staticmethod
    def get_bounds_from_dataset(
        dataset: gdal.Dataset,
        crs: pyproj.CRS = None,
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
        x_bounds = [x_min, x_max]
        y_bounds = [y_min, y_max]

        # Convert to desired crs.
        dataset_crs = pyproj.CRS(dataset.GetProjection())
        if crs is None:
            crs = dataset_crs
        else:
            dataset_to_desired = pyproj.Transformer.from_crs(
                dataset_crs,
                crs,
                always_xy=True
            )
            x_bounds, y_bounds = dataset_to_desired.transform(
                x_bounds,
                y_bounds,
            )
            pixel_width, pixel_height = dataset_to_desired.transform(
                pixel_width,
                pixel_height,
            )

        return (
            x_bounds,
            y_bounds,
            pixel_width,
            pixel_height,
            crs
        )


class RegisteredImageIO(DataIO):
    name = 'registered_image'

    @staticmethod
    def save(filepath, img, x_bounds, y_bounds, crs, driver='GTiff'):

        if filepath != '':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Get data type
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(img.dtype)

        # Create dataset
        driver_obj = gdal.GetDriverByName(driver)
        dataset = driver_obj.Create(
            filepath,
            img.shape[1],
            img.shape[0],
            img.shape[2],
            gdal_dtype,
        )

        # Write to the dataset
        dataset.WriteArray(img.transpose(2, 0, 1))

        # Set CRS properties
        dataset.SetProjection(crs.to_wkt())

        # Set geotransform
        dx = (x_bounds[1] - x_bounds[0]) / img.shape[1]
        dy = (y_bounds[1] - y_bounds[0]) / img.shape[0]
        geotransform = (
            x_bounds[0],
            dx,
            0,
            y_bounds[1],
            0,
            -dy
        )
        dataset.SetGeoTransform(geotransform)

        # Stop and return if we want to keep the dataset open
        if driver == 'MEM':
            return dataset

        dataset.FlushCache()
        dataset = None

    @staticmethod
    def load(filepath, crs: pyproj.CRS = None):

        # Get image
        dataset = GDALDatasetIO.load(filepath)
        img = dataset.ReadAsArray().transpose(1, 2, 0)

        # Get bounds
        (
            x_bounds, y_bounds, dx, dy, crs
        ) = GDALDatasetIO.get_bounds_from_dataset(
            dataset=dataset,
            crs=crs,
        )

        data = (img, x_bounds, y_bounds, crs)

        return data


class CSVIO(DataIO):
    name = 'csv'

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
