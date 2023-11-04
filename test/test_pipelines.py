'''Test file for pipelines.
'''

import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

import night_horizons_mapmaker.preprocess as preprocess
import night_horizons_mapmaker.pipelines as pipelines


class TestPreprocess(unittest.TestCase):

    def setUp(self):

        self.pipelines = pipelines.PreprocessingPipelines()

    def test_referenced_nitelite(self):

        image_dir = './test/test_data/referenced_images'
        fps = preprocess.discover_data(image_dir)

        # Metadata filetree info
        metadata_dir = './test/test_data/metadata'
        img_log_fp = os.path.join(metadata_dir, 'image.log')
        imu_log_fp = os.path.join(metadata_dir, 'PresIMULog.csv')
        gps_log_fp = os.path.join(metadata_dir, 'GPSLog.csv')

        ref_nitelite_pipeline = self.pipelines.referenced_nitelite()

        output_df = ref_nitelite_pipeline.fit_transform(
            fps,
            nitelite__img_log_fp=img_log_fp,
            nitelite__imu_log_fp=imu_log_fp,
            nitelite__gps_log_fp=gps_log_fp,
        )
        expected_cols = (
            preprocess.GEOTRANSFORM_COLS
            + ['filepath', 'sensor_x', 'sensor_y', ]
        )
        matching_cols = output_df.columns.isin(expected_cols)
        assert matching_cols.sum() == len(expected_cols)
        assert (~matching_cols).sum() == 0


class TestGeoreference(unittest.TestCase):

    def setUp(self):

        self.pipelines = pipelines.GeoreferencingPipelines()

    def test_sensor_georeferencing(self):

        image_dir = './test/test_data/referenced_images'
        fps = preprocess.discover_data(image_dir)

        # Metadata filetree info
        metadata_dir = './test/test_data/metadata'
        img_log_fp = os.path.join(metadata_dir, 'image.log')
        imu_log_fp = os.path.join(metadata_dir, 'PresIMULog.csv')
        gps_log_fp = os.path.join(metadata_dir, 'GPSLog.csv')

        # Get the y values
        y_pipeline = pipelines.PreprocessingPipelines.referenced_nitelite()
        y = y_pipeline.fit_transform(
            fps,
            nitelite__img_log_fp=img_log_fp,
            nitelite__imu_log_fp=imu_log_fp,
            nitelite__gps_log_fp=gps_log_fp,
        )

        # Train test split
        inds_train, inds_test = train_test_split(
            fps.index,
            test_size=0.2,
        )
        fps_train = fps.loc[inds_train]
        fps_test = fps.loc[inds_test]
        y_train = y.loc[inds_train]
        y_test = y.loc[inds_test]

        sensor_ref_pipeline = self.pipelines.sensor_georeferencing()
        sensor_ref_pipeline.fit(
            fps_train,
            preprocessing__nitelite__img_log_fp=img_log_fp,
            preprocessing__nitelite__imu_log_fp=imu_log_fp,
            preprocessing__nitelite__gps_log_fp=gps_log_fp,
        )
        y_pred = sensor_ref_pipeline.predict(fps_test)
        expected_cols = (
            preprocess.GEOTRANSFORM_COLS
            + ['estimated_spatial_error',]
        )
        matching_cols = y_pred.columns.isin(expected_cols)
        assert matching_cols.sum() == len(expected_cols)
        assert (~matching_cols).sum() == 0
