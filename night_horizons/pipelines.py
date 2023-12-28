'''Pipelines for various purposes.
'''

from typing import Union

import numpy as np
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .processors import image_registration, mosaickers
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import preprocessors


class PreprocessingPipelines:

    @staticmethod
    def nitelite_preprocessing_steps(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        use_approximate_georeferencing: bool = True,
        altitude_column: str = 'mAltitude',
        gyro_columns: list[str] = ['imuGyroX', 'imuGyroY', 'imuGyroZ'],
        padding_fraction: float = 0.5,
    ):
        '''
        TODO: Remove parameters from nitelite_preprocessing.
        This should be a ready-to-go pipeline. If the user really wants to
        tweak parameters it should either be done in post or they should
        make their own pipeline.

        Parameters
        ----------
        Returns
        -------
        '''

        # Choose the preprocessor for getting the bulk of the metadata
        metadata_preprocessor = preprocessors.NITELitePreprocessor(crs=crs)

        # Choose the georeferencing
        if use_approximate_georeferencing:
            georeferencer = image_registration.MetadataImageRegistrar(
                crs=crs,
                passthrough=['filepath', 'camera_num'],
                padding_fraction=padding_fraction,
            )
        else:
            georeferencer = preprocessors.GeoTIFFPreprocessor(
                crs=crs,
                passthrough=['camera_num'],
                padding_fraction=padding_fraction,
            )

        # Build the steps
        preprocessing_steps = [
            ('metadata',
             metadata_preprocessor),
            ('select_deployment_phase',
             preprocessors.AltitudeFilter(column=altitude_column)),
            ('select_steady',
             preprocessors.SteadyFilter(columns=gyro_columns)),
            ('georeference', georeferencer),
            ('order', preprocessors.SensorAndDistanceOrder()),
        ]

        return preprocessing_steps


class GeoreferencePipelines:

    @staticmethod
    def sensor_georeferencing(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        # Choose the preprocessor for getting the bulk of the metadata
        metadata_preprocessor = preprocessors.NITELitePreprocessor(
            crs=crs,
            unhandled_files='warn and drop',
        )
        pipeline = Pipeline([
            ('nitelite', metadata_preprocessor),
            ('georeference', image_registration.MetadataImageRegistrar(crs=crs)),
        ])

        return pipeline


class MosaicPipelines:

    @staticmethod
    def referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('geotiff', preprocessors.GeoTIFFPreprocessor(crs=crs)),
            ('mosaic', mosaickers.Mosaicker(filepath=filepath, crs=crs))
        ])

        return pipeline

    @staticmethod
    def less_referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocessors.NITELitePreprocessor(
                output_columns=['filepath', 'sensor_x', 'sensor_y'])),
            ('geotiff', preprocessors.GeoTIFFPreprocessor(
                crs=crs, passthrough=True)),
            ('mosaic', mosaickers.SequentialMosaicker(filepath=filepath, crs=crs))
        ])

        return pipeline
