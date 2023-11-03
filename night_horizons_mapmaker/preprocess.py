import glob
import os
from typing import Union

import numpy as np
from numpy import ndarray
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MetadataPreprocesser(TransformerMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.

    Parameters
    ----------
    output_columns :
        What columns to include in the output.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, output_columns: list[str] = None):
        self.output_columns = output_columns

    # def fit(self, X, y=None):
    #     """A reference implementation of a fitting function for a transformer.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.
    #     y : None
    #         There is no need of a target in a transformer, yet the pipeline API
    #         requires this parameter.

    #     Returns
    #     -------
    #     self : object
    #         Returns self.
    #     """
    #     X = check_array(X, dtype='str')

    #     self.n_features_ = X.shape[1]

    #     # Return the transformer
    #     return self

    # def transform(self, X):
    #     """ A reference implementation of a transform function.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse-matrix}, shape (n_samples, n_features)
    #         The input samples.

    #     Returns
    #     -------
    #     X_transformed : array, shape (n_samples, n_features)
    #         The array containing the element-wise square roots of the values
    #         in ``X``.
    #     """
    #     # Check is fit had been called
    #     check_is_fitted(self, 'n_features_')

    #     # Input validation
    #     X = check_array(X, dtype='str')

    #     # check that the input is of the same shape as the one passed
    #     # during fit.
    #     if X.shape[1] != self.n_features_:
    #         raise ValueError('Shape of input is different from what was seen'
    #                          'in `fit`')
    #     return np.sqrt(X)

    def fit_transform(
        self,
        X: np.ndarray[str],
        y=None,
        img_log_fp: str = None,
        imu_log_fp: str = None,
        gps_log_fp: str = None,
    ):

        assert img_log_fp is not None, 'Must pass img_log filepath.'
        assert imu_log_fp is not None, 'Must pass imu_log filepath.'
        assert gps_log_fp is not None, 'Must pass gps_log filepath.'

        X = check_array(X, dtype='str')

        # Unpack X (only fps inside)
        fps = X[0]

        # Get the raw metadata
        log_df = self.get_all_logs(img_log_fp, imu_log_fp, gps_log_fp)

        metadata_df = self.correlate_filepaths(fps, log_df)

    def correlate_filepaths(
        self,
        filepaths: np.ndarray[str],
        log_df: pd.DataFrame,
    ):

        pass

    def get_all_logs(
        self,
        img_log_fp: str,
        imu_log_fp: str,
        gps_log_fp: str,
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
                - pd.Timedelta(self.metadata_tz_offset_in_hr, 'hr')
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
        def get_filepath(filename):
            return os.path.join(
                self.image_dir,
                filename,
            )
        img_log_df['obc_filename'] = img_log_df['filename'].copy()
        img_log_df['filename'] = img_log_df['obc_filename'].apply(
            os.path.basename
        )
        img_log_df['filepath'] = img_log_df['filename'].apply(get_filepath)
        # TODO: Currently flight only deals with one camera at a time.
        # Filepaths that are for other cameras are invalid.
        img_log_df['valid_filepath'] = img_log_df['filename'].isin(
            os.listdir(self.image_dir)
        )

        self.img_log_df = img_log_df
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

        self.imu_log_df = imu_log_df
        return imu_log_df

    def load_gps_log(self, gps_log_fp: str = None) -> pd.DataFrame:
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
        gps_log_df['sensor_x'], gps_log_df['sensor_y'] = \
            self.latlon_to_cart.transform(
                gps_log_df['GPSLat'],
                gps_log_df['GPSLong']
        )

        self.gps_log_df = gps_log_df
        return gps_log_df




def discover_data(
    directory: str,
    extension: Union[str, list[str]] = None
) -> list[str]:
    '''
    Parameters
    ----------
        directory:
            Directory containing the data.
        extension:
            What filetypes to include.

    Returns
    -------
        filepaths:
            Data filepaths.
    '''

    # When all files
    if extension is None:
        pattern = os.path.join(directory, '**', '*.*')
        return glob.glob(pattern, recursive=True)
    # When a single extension
    elif isinstance(extension, str):
        pattern = os.path.join(directory, '**', f'*.{extension}')
        return glob.glob(pattern, recursive=True)
    # When a list of extensions
    else:
        try:
            fps = []
            for ext in extension:
                pattern = os.path.join(directory, '**', f'*.{ext}')
                fps.extend(glob.glob(pattern, recursive=True))
            return fps
        except TypeError:
            raise TypeError(f'Unexpected type for extension: {extension}')
