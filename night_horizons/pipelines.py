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

from . import preprocess, reference, mosaic


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
        metadata_preprocesser = preprocess.NITELitePreprocesser(crs=crs)

        # Choose the georeferencing
        if use_approximate_georeferencing:
            georeferencer = reference.SensorGeoreferencer(
                crs=crs,
                passthrough=['filepath', 'camera_num'],
                padding_fraction=padding_fraction,
            )
        else:
            georeferencer = preprocess.GeoTIFFPreprocesser(
                crs=crs,
                passthrough=['camera_num'],
                padding_fraction=padding_fraction,
            )

        # Build the steps
        preprocessing_steps = [
            ('metadata',
             metadata_preprocesser),
            ('select_deployment_phase',
             preprocess.AltitudeFilter(column=altitude_column)),
            ('select_steady',
             preprocess.SteadyFilter(columns=gyro_columns)),
            ('georeference', georeferencer),
            ('order', preprocess.SensorAndDistanceOrder()),
        ]

        return preprocessing_steps


class GeoreferencePipelines:

    @staticmethod
    def sensor_georeferencing(
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        # Choose the preprocesser for getting the bulk of the metadata
        metadata_preprocesser = preprocess.NITELitePreprocesser(
            crs=crs,
            unhandled_files='warn and drop',
        )
        pipeline = Pipeline([
            ('nitelite', metadata_preprocesser),
            ('georeference', reference.SensorGeoreferencer(crs=crs)),
        ])

        return pipeline


class MosaicPipelines:

    @staticmethod
    def referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('geotiff', preprocess.GeoTIFFPreprocesser(crs=crs)),
            ('mosaic', mosaic.ReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline

    @staticmethod
    def less_referenced_mosaic(
        filepath: str,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
    ):

        pipeline = Pipeline([
            ('nitelite', preprocess.NITELitePreprocesser(
                output_columns=['filepath', 'sensor_x', 'sensor_y'])),
            ('geotiff', preprocess.GeoTIFFPreprocesser(
                crs=crs, passthrough=True)),
            ('mosaic', mosaic.LessReferencedMosaic(filepath=filepath, crs=crs))
        ])

        return pipeline
