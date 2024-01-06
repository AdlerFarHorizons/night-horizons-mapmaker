'''TODO: Finish implementing this.
'''
from abc import ABC, abstractmethod
import inspect
import pickle

import cv2
from osgeo import gdal, gdal_array
import pandas as pd
import yaml


class DataIO(ABC):
    @abstractmethod
    def save_data(self, data, filepath):
        pass

    @abstractmethod
    def load_data(self, filepath):
        pass


class RegisteredImageIO(DataIO):
    name = 'registered_image'

    def __init__(self, crs):
        self.crs = crs

    def save_data(self, data, filepath, x_min, x_max, y_min, y_max):

        # Get data type
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

        # Create dataset
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
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
        self.dataset.SetGeoTransform(geotransform)

        dataset.FlushCache()
        dataset = None

    def load_data(self, filepath):
        dataset = gdal.Open(filepath)
        data = dataset.GetRasterBand(1).ReadAsArray()
        return data


class GDALDatasetIO(DataIO):
    name = 'gdal_dataset'

    def __init__(self, crs):
        self.crs = crs

    def save_data(self, data, filepath):
        pass

    def load_data(self, filepath):
        data = gdal.Open(filepath, gdal.GA_ReadOnly)
        data.SetProjection(self.crs.to_wkt())
        return data


class ImageDataIO(DataIO):
    name = 'image'

    def save_data(self, data, filepath):
        cv2.imwrite(filepath, data)

    def load_data(self, filepath):
        data = cv2.imread(filepath)
        return data


class TabularDataIO(DataIO):
    name = 'tabular'

    def save_data(self, data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df


class YAMLDataIO(DataIO):
    name = 'yaml'

    def save_data(self, data, filepath):

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

    def load_data(self, filepath):
        pass
