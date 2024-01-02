from abc import ABC, abstractmethod
import inspect
import pickle

import cv2
from osgeo import gdal
import pandas as pd


class DataIO(ABC):
    @abstractmethod
    def save_data(self, data, filepath):
        pass

    @abstractmethod
    def load_data(self, filepath):
        pass


class GDALDataIO(DataIO):
    def save_data(self, data, filepath):
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filepath, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        dataset.GetRasterBand(1).WriteArray(data)
        dataset.FlushCache()

    def load_data(self, filepath):
        dataset = gdal.Open(filepath)
        data = dataset.GetRasterBand(1).ReadAsArray()
        return data


class ImageDataIO(DataIO):
    def save_data(self, data, filepath):
        cv2.imwrite(filepath, data)

    def load_data(self, filepath):
        data = cv2.imread(filepath)
        return data


class TabularDataIO(DataIO):
    def save_data(self, data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        data = df.values
        return data


class YAMLDataIO(DataIO):
    def save_data(self, data, filepath):

        fullargspec = inspect.getfullargspec(type(obj))
        settings = {}
        for setting in fullargspec.args:
            if setting == 'self':
                continue
            value = getattr(obj, setting)
            try:
                pickle.dumps(value)
            except TypeError:
                value = 'no string repr'
            settings[setting] = value
        with open(self.output_filepaths['settings'], 'w') as file:
            yaml.dump(settings, file)

    def load_data(self, filepath):
        pass
