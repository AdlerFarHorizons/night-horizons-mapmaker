import glob
import os
from typing import Union
import warnings

import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from . import utils

GEOTRANSFORM_COLS = [
    'x_min', 'x_max',
    'y_min', 'y_max',
    'pixel_width', 'pixel_height',
    'x_rot', 'y_rot',
    'x_size', 'y_size',
    'x_center', 'y_center',
]


class NITELitePreprocesser(TransformerMixin, BaseEstimator):
    '''Transform filepaths into a metadata dataframe.

    Parameters
    ----------
    output_columns :
        What columns to include in the output.

    crs:
        The coordinate reference system to use.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    '''

    def __init__(
        self,
        output_columns: list[str] = None,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        unhandled_files: str = 'warn and passthrough',
        passthrough: list[str] = [],
    ):
        self.output_columns = output_columns
        self.crs = crs
        self.unhandled_files = unhandled_files
        self.passthrough = passthrough

    def fit(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
        img_log_fp: str = None,
        imu_log_fp: str = None,
        gps_log_fp: str = None,
    ):

        # Check the input is good.
        X = utils.check_filepaths_input(
            X,
            passthrough=self.passthrough,
        )

        # Convert CRS as needed
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        # Check and set log fps
        assert img_log_fp is not None, 'Must pass img_log filepath.'
        assert imu_log_fp is not None, 'Must pass imu_log filepath.'
        assert gps_log_fp is not None, 'Must pass gps_log filepath.'
        self.img_log_fp_ = img_log_fp
        self.imu_log_fp_ = imu_log_fp
        self.gps_log_fp_ = gps_log_fp

        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ):

        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Check the input is good.
        X = utils.check_filepaths_input(
            X,
        )

        # Get the raw metadata
        log_df = self.get_logs(
            self.img_log_fp_,
            self.imu_log_fp_,
            self.gps_log_fp_,
        )

        # Merge, assuming filenames remain the same.
        X['original_index'] = X.index
        X['filename'] = X['filepath'].apply(os.path.basename)
        X_corr = pd.merge(
            X,
            log_df,
            how='inner',
            on='filename'
        )
        # Leftovers
        X_remain = (X.loc[~X.index.isin(X_corr['original_index'])]).copy()

        if len(X_remain) > 0:
            # Secondary merge attempt, using a common pattern
            pattern = r'(\d+)_\d.tif'
            X_remain['timestamp_id'] = X_remain['filename'].str.findall(
                pattern
            ).str[-1].astype('Int64')
            X_corr2 = pd.merge(
                X_remain,
                log_df,
                how='inner',
                on='timestamp_id'
            )

            # Recombine
            X_out = pd.concat([X_corr, X_corr2], axis='rows')
        else:
            X_out = X_corr

        # At the end, what are we still missing?
        is_missing = ~X.index.isin(X_out['original_index'])
        n_uncorrelated = is_missing.sum()
        w_message = (
            'Did not successfully correlate all filepaths. '
            f'n_uncorrelated = {n_uncorrelated}'
        )
        if n_uncorrelated > 0:
            if self.unhandled_files == 'error':
                assert False, w_message
            elif 'drop' in self.unhandled_files:
                if 'warn' in self.unhandled_files:
                    warnings.warn(w_message)
                pass
            elif 'passthrough' in self.unhandled_files:
                if 'warn' in self.unhandled_files:
                    warnings.warn(w_message)
                X_missing = X.loc[is_missing]
                X_out = pd.concat([X_out, X_missing])
            else:
                raise ValueError('Unrecognized method for unhandled files.')

        # Organize the index
        X_out.set_index('original_index', inplace=True)
        # Don't try to sort by indices X_out does not have
        sort_inds = X.index[X.index.isin(X_out.index)]
        X_out = X_out.loc[sort_inds]

        # Select only the desired columns
        if self.output_columns is not None:
            X_out = X_out[self.output_columns]

        return X_out

    def get_logs(
        self,
        img_log_fp: str,
        imu_log_fp: str,
        gps_log_fp: str,
        tz_offset_in_hr: float = 5.,
    ) -> pd.DataFrame:
        '''Combine the different logs

        Parameters
        ----------
            img_log_df:
                DataFrame containing image metadata.
            imu_log_df:
                DataFrame containing IMU metadata.
            gps_log_df:
                DataFrame containing GPS metadata.

        Returns
        -------
            log_df:
                Combined dataframe containing IMU and GPS metadata
                for each image.
        '''

        img_log_df = self.load_img_log(img_log_fp)
        imu_log_df = self.load_imu_log(imu_log_fp)
        gps_log_df = self.load_gps_log(gps_log_fp)

        dfs_interped = [img_log_df, ]
        source_log_names = ['imu', 'gps']
        for i, df_to_include in enumerate([imu_log_df, gps_log_df]):

            source_log_name = source_log_names[i]
            df_to_include = df_to_include.copy()

            # This doesn't interpolate well unless converted
            if 'GPSTime' in df_to_include.columns:
                del df_to_include['GPSTime']

            # Get the timestamps in the right time zone
            df_to_include['CurrTimestamp_in_img_tz'] = (
                df_to_include['CurrTimestamp']
                - pd.Timedelta(tz_offset_in_hr, 'hr')
            )
            df_to_include = df_to_include.dropna(
                subset=['CurrTimestamp_in_img_tz']
            ).set_index('CurrTimestamp_in_img_tz').sort_index()
            df_to_include['timestamp_int_{}'.format(source_log_name)] = \
                df_to_include['CurrTimestamp'].astype(int)
            del df_to_include['CurrTimestamp']

            # Interpolate
            interp_fn = scipy.interpolate.interp1d(
                df_to_include.index.astype(int),
                df_to_include.values.transpose()
            )
            interped = interp_fn(img_log_df['timestamp'].astype(int))
            df_interped = pd.DataFrame(
                interped.transpose(),
                columns=df_to_include.columns
            )

            dfs_interped.append(df_interped)

        log_df = pd.concat(dfs_interped, axis='columns', )

        return log_df

    def load_img_log(self, img_log_fp: str = None) -> pd.DataFrame:
        '''Load the images log.

        Parameters
        ----------
            img_log_fp: Location of the image log.
                Defaults to the one provided at init.
        '''

        # Load data
        # Column names are known and input ad below.
        img_log_df = pd.read_csv(
            img_log_fp,
            names=[
                'odroid_timestamp',
                'obc_timestamp',
                'camera_num',
                'serial_num',
                'exposure_time',
                'sequence_ind',
                'internal_temp',
                'filename',
            ] + ['Unnamed: {}'.format(i + 1) for i in range(12)]
        )

        # Parse the timestamp
        # We use a combination of the odroid timestamp and the obc
        # timestamp because the odroid timestamp is missing the year but
        # the obc_timestamp has the wrong month.
        timestamp_split = img_log_df['obc_timestamp'].str.split('_')
        img_log_df['obc_timestamp'] = pd.to_datetime(
            timestamp_split.apply(lambda x: '_'.join(x[:2])),
            format=' %Y%m%d_%H%M%S'
        )
        img_log_df['timestamp'] = pd.to_datetime(
            img_log_df['obc_timestamp'].dt.year.astype(str)
            + ' '
            + img_log_df['odroid_timestamp']
        )
        img_log_df['timestamp_id'] = timestamp_split.apply(
            lambda x: x[-1]
        ).astype(int)
        # Correct for overflow
        img_log_df.loc[img_log_df['timestamp_id'] < 0, 'timestamp_id'] += 2**32

        # Drop unnamed columns
        img_log_df = img_log_df.drop(
            [column for column in img_log_df.columns if 'Unnamed' in column],
            axis='columns'
        )

        # Get filepaths
        img_log_df['obc_filename'] = img_log_df['filename'].copy()
        img_log_df['filename'] = img_log_df['obc_filename'].apply(
            os.path.basename
        )

        return img_log_df

    def load_imu_log(self, imu_log_fp: str = None) -> pd.DataFrame:
        '''Load the IMU log.

        Args:
            imu_log_fp: Location of the IMU log.
                Defaults to the one provided at init.
        '''

        if imu_log_fp is None:
            imu_log_fp = self.imu_log_fp

        imu_log_df = pd.read_csv(imu_log_fp, low_memory=False)

        # Remove the extra header rows, and the nan rows
        imu_log_df.dropna(subset=['CurrTimestamp', ], inplace=True)
        imu_log_df.drop(
            imu_log_df.index[imu_log_df['CurrTimestamp'] == 'CurrTimestamp'],
            inplace=True
        )

        # Handle some situations where the pressure is negative
        ac_columns = ['TempC', 'pressure', 'mAltitude']
        imu_log_df.loc[imu_log_df['pressure'].astype(float) < 0] = np.nan

        # Convert to datetime and sort
        imu_log_df['CurrTimestamp'] = pd.to_datetime(
            imu_log_df['CurrTimestamp']
        )
        imu_log_df.sort_values('CurrTimestamp', inplace=True)

        # Assign dtypes
        skipped_cols = []
        for column in imu_log_df.columns:
            if column == 'CurrTimestamp':
                continue

            imu_log_df[column] = imu_log_df[column].astype(float)

        # Now also handle when the altitude or temperature are weird
        imu_log_df.loc[imu_log_df['TempC'] < -273, ac_columns] = np.nan
        imu_log_df.loc[imu_log_df['mAltitude'] < 0., ac_columns] = np.nan
        imu_log_df.loc[imu_log_df['mAltitude'] > 20000., ac_columns] = np.nan

        return imu_log_df

    def load_gps_log(
        self,
        gps_log_fp: str = None,
        latlon_crs: str = 'EPSG:4326'
    ) -> pd.DataFrame:
        '''Load the GPS log.

        Args:
            gps_log_fp: Location of the GPS log.
                Defaults to the one provided at init.
        '''

        if gps_log_fp is None:
            gps_log_fp = self.gps_log_fp

        gps_log_df = pd.read_csv(gps_log_fp)

        # Remove the extra header rows and the empty rows
        gps_log_df.dropna(subset=['CurrTimestamp', ], inplace=True)
        gps_log_df.drop(
            gps_log_df.index[gps_log_df['CurrTimestamp'] == 'CurrTimestamp'],
            inplace=True
        )

        # Remove the empty rows
        empty_timestamp = '00.00.0000 00:00:00000'
        gps_log_df.drop(
            gps_log_df.index[gps_log_df['CurrTimestamp'] == empty_timestamp],
            inplace=True
        )

        # Convert to datetime and sort
        gps_log_df['CurrTimestamp'] = pd.to_datetime(
            gps_log_df['CurrTimestamp']
        )
        gps_log_df.sort_values('CurrTimestamp', inplace=True)

        # Assign dtypes
        for column in gps_log_df.columns:
            if column in ['CurrTimestamp', 'GPSTime']:
                continue

            gps_log_df[column] = gps_log_df[column].astype(float)

        # Coordinates
        if isinstance(latlon_crs, str):
            latlon_crs = pyproj.CRS(latlon_crs)
        latlon_crs_to_crs = pyproj.Transformer.from_crs(
            latlon_crs,
            self.crs
        )
        gps_log_df['sensor_x'], gps_log_df['sensor_y'] = \
            latlon_crs_to_crs.transform(
                gps_log_df['GPSLat'],
                gps_log_df['GPSLong']
        )

        return gps_log_df


class GeoTIFFPreprocesser(TransformerMixin, BaseEstimator):
    '''Transform filepaths into geotransform properties.

    Parameters
    ----------
    output_columns :
        What columns to include in the output.

    crs:
        The coordinate reference system to use.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    '''

    def __init__(
        self,
        crs: Union[str, pyproj.CRS] = 'EPSG:3857',
        passthrough: bool = False,
    ):
        self.crs = crs
        self.passthrough = passthrough

    def fit(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ):

        X = utils.check_filepaths_input(X, passthrough=self.passthrough)

        # Convert CRS as needed
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ):

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Loop over and get datasets
        rows = []
        for i, fp in enumerate(tqdm.tqdm(X['filepath'], ncols=80)):

            # Try to load the dataset
            dataset = gdal.Open(fp, gdal.GA_ReadOnly)
            if dataset is None:
                row = pd.Series(
                    [np.nan,] * len(GEOTRANSFORM_COLS),
                    index=GEOTRANSFORM_COLS,
                    name=X.index[i]
                )
                rows.append(row)
                continue

            x_min, pixel_width, x_rot, y_max, y_rot, pixel_height = \
                dataset.GetGeoTransform()

            # Get bounds
            x_max = x_min + pixel_width * dataset.RasterXSize
            y_min = y_max + pixel_height * dataset.RasterYSize

            # Convert to desired crs.
            dataset_crs = pyproj.CRS(dataset.GetProjection())
            dataset_crs_to_crs = pyproj.Transformer.from_crs(
                dataset_crs,
                self.crs,
                always_xy=True
            )
            (x_min, x_max), (y_min, y_max) = dataset_crs_to_crs.transform(
                [x_min, x_max],
                [y_min, y_max],
            )
            pixel_width, pixel_height = dataset_crs_to_crs.transform(
                pixel_width,
                pixel_height,
            )

            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)

            row = pd.Series(
                [
                    x_min, x_max,
                    y_min, y_max,
                    pixel_width, pixel_height,
                    x_rot, y_rot,
                    dataset.RasterXSize, dataset.RasterYSize,
                    x_center, y_center
                ],
                index=GEOTRANSFORM_COLS,
                name=X.index[i]
            )
            rows.append(row)

        new_df = pd.DataFrame(rows)

        X = pd.concat([X, new_df], axis='columns')

        return X
