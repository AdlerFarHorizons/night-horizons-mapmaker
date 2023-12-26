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
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from . import image_registration, mosaickers, preprocessers


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

        # Choose the preprocesser for getting the bulk of the metadata
        metadata_preprocesser = preprocessers.NITELitePreprocesser(crs=crs)

        # Choose the georeferencing
        if use_approximate_georeferencing:
            georeferencer = image_registration.MetadataImageRegistrar(
                crs=crs,
                passthrough=['filepath', 'camera_num'],
                padding_fraction=padding_fraction,
            )
        else:
            georeferencer = preprocessers.GeoTIFFPreprocesser(
                crs=crs,
                passthrough=['camera_num'],
                padding_fraction=padding_fraction,
            )

        # Build the steps
        preprocessing_steps = [
            ('metadata',
             metadata_preprocesser),
            ('select_deployment_phase',
             preprocessers.AltitudeFilter(column=altitude_column)),
            ('select_steady',
             preprocessers.SteadyFilter(columns=gyro_columns)),
            ('georeference', georeferencer),
            ('order', preprocessers.SensorAndDistanceOrder()),
        ]

        return preprocessing_steps


class GeoreferencePipelines:

    @staticmethod
    def sensor_georeferencing(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        # Choose the preprocesser for getting the bulk of the metadata
        metadata_preprocesser = preprocessers.NITELitePreprocesser(
            crs=crs,
            unhandled_files='warn and drop',
        )
        pipeline = Pipeline([
            ('nitelite', metadata_preprocesser),
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
            ('geotiff', preprocessers.GeoTIFFPreprocesser(crs=crs)),
            ('mosaic', mosaickers.ReferencedMosaicker(filepath=filepath, crs=crs))
        ])

        return pipeline

    @staticmethod
    def less_referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocessers.NITELitePreprocesser(
                output_columns=['filepath', 'sensor_x', 'sensor_y'])),
            ('geotiff', preprocessers.GeoTIFFPreprocesser(
                crs=crs, passthrough=True)),
            ('mosaic', mosaickers.LessReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline
