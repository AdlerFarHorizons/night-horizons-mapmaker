'''TODO: Finish implementing this.
'''
from abc import ABC, abstractmethod
import inspect
import os
import pickle

import cv2
from osgeo import gdal, gdal_array
import pandas as pd
import pyproj
import yaml


class DataIO(ABC):
    @abstractmethod
    def save(self, data, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class ImageIO(DataIO):
    name = 'image'

    @staticmethod
    def save(data, filepath):

        data = data[:, :, ::-1]
        cv2.imwrite(filepath, data)

    @staticmethod
    def load(filepath):

        data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        data = data[:, :, ::-1]

        return data


class RegisteredImageIO(DataIO):
    name = 'registered_image'

    @staticmethod
    def save(data, filepath, x_min, x_max, y_min, y_max, crs, driver='GTiff'):

        if filepath != '':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Get data type
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

        # Create dataset
        driver_obj = gdal.GetDriverByName(driver)
        dataset = driver_obj.Create(
            filepath,
            data.shape[1],
            data.shape[0],
            data.shape[2],
            gdal_dtype,
        )

        # Write to the dataset
        dataset.WriteArray(data.transpose(2, 0, 1))

        # Set CRS properties
        dataset.SetProjection(crs.to_wkt())

        # Set geotransform
        dx = (x_max - x_min) / data.shape[1]
        dy = (y_max - y_min) / data.shape[0]
        geotransform = (
            x_min,
            dx,
            0,
            y_max,
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
    def load(filepath):
        dataset = gdal.Open(filepath)
        data = dataset.ReadAsArray()
        return data


class GDALDatasetIO(DataIO):
    name = 'gdal_dataset'

    @staticmethod
    def save(data, filepath):
        pass

    @staticmethod
    def load(self, filepath):
        data = gdal.Open(filepath, gdal.GA_ReadOnly)
        if self.crs is not None:
            data.SetProjection(self.crs.to_wkt())
        return data


class TabularIO(DataIO):
    name = 'tabular'

    @staticmethod
    def save(data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath):
        df = pd.read_csv(filepath)
        return df


class YAMLIO(DataIO):
    name = 'yaml'

    @staticmethod
    def save(self, data, filepath):

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
        with open(self.output_filepaths['settings'], 'w') as file:
            yaml.dump(settings, file)

    @staticmethod
    def load(self, filepath):
        pass
