"""Module for miscellaneous preprocessing of the data."""

import os
from typing import Union
import warnings

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
import pandas as pd
import pyproj
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import io_manager, utils

GEOTRANSFORM_COLS = [
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "pixel_width",
    "pixel_height",
    "x_rot",
    "y_rot",
    "x_size",
    "y_size",
    "x_center",
    "y_center",
    "spatial_error",
    "padding",
]


class MetadataPreprocessor(TransformerMixin, BaseEstimator):
    """Class for transforming filepaths into a metadata dataframe."""

    def __init__(
        self,
        io_manager: io_manager.IOManager,
        crs: pyproj.CRS,
        output_columns: list[str] = None,
        use_cached_output: bool = True,
        unhandled_files: str = "drop",
        tz_offset_in_hr: float = 5.0,
        passthrough: list[str] = [],
    ):
        """
        Initialize the MetadataPreprocessor object.

        Parameters
        ----------
        io_manager : io_manager.IOManager
            The IOManager object used for input/output operations.
        crs : pyproj.CRS
            The coordinate reference system (CRS) to be used for the data.
        output_columns : list[str], optional
            The list of output column names, by default None, which keeps all
            the new columns retrieved.
        use_cached_output : bool, optional
            Flag indicating whether to use cached output, by default True.
        unhandled_files : str, optional
            Action to be taken for unhandled files, by default 'drop'.
        passthrough : list[str], optional
            List of column names to be passed through without any processing,
            by default [].
        tz_offset_in_hr : float, optional
            Timezone offset in hours, by default 5.
        """
        self.io_manager = io_manager
        self.crs = crs
        self.output_columns = output_columns
        self.required_columns = ["filepath"]
        self.use_cached_output = use_cached_output
        self.unhandled_files = unhandled_files
        self.tz_offset_in_hr = tz_offset_in_hr
        self.passthrough = passthrough

    def fit(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ) -> "MetadataPreprocessor":
        """
        Fits the MetadataPreprocessor to the input data.

        Parameters
        ----------
        X : Union[np.ndarray[str], list[str], pd.DataFrame]
            The input filepaths to fit the preprocessor on.

        y : None, optional
            The target variable (default is None).

        Returns
        -------
        MetadataPreprocessor
            The fitted MetadataPreprocessor object.
        """

        # Check the input is good.
        X = utils.check_filepaths_input(
            X,
            passthrough=self.passthrough,
        )

        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ) -> pd.DataFrame:
        """
        Transform the input data by performing preprocessing steps for metadata.

        Parameters
        ----------
        X : Union[np.ndarray[str], list[str], pd.DataFrame]
            The input filepaths to be transformed.
        y : None, optional
            The target variable, if applicable. Default is None.

        Returns
        -------
        pd.DataFrame
            The transformed data as a DataFrame.
        """

        # Check if fit had been called
        check_is_fitted(self, "is_fitted_")

        # Check the input is good.
        X = utils.check_filepaths_input(
            X,
        )

        # Try loading existing data.
        try:
            assert self.use_cached_output, "Not using cached output."
            X_out = pd.read_csv(
                self.io_manager.input_filepaths["metadata"],
                index_col=0,
            )
        except (KeyError, FileNotFoundError, AssertionError) as _:

            # Do the calculations
            log_df = self.get_logs()

            # Merge, assuming filenames remain the same.
            X["original_index"] = X.index
            X["filename"] = X["filepath"].apply(os.path.basename)
            X_out = pd.merge(X, log_df, how="inner", on="filename")

            # Rename filepath_x
            if "filepath_x" in X_out.columns:
                X_out["filepath"] = X_out["filepath_x"]
                del X_out["filepath_x"]

            # Try a few patterns to see if we can get the rest
            patterns = [r"(\d+)_\d.tif", r"(\d+)\.\d+.tif"]
            for pattern in patterns:
                # Leftovers
                X_remain = (X.loc[~X.index.isin(X_out["original_index"])]).copy()

                if len(X_remain) > 0:
                    # Secondary merge attempt, using a common pattern
                    X_remain["timestamp_id"] = (
                        X_remain["filename"].str.findall(pattern).str[-1]
                    )
                    X_remain = X_remain.dropna(subset=["timestamp_id"])
                    X_out2 = pd.merge(X_remain, log_df, how="inner", on="timestamp_id")

                    # Recombine
                    X_out = pd.concat([X_out, X_out2], axis="rows")

            X_out.set_index("original_index", inplace=True)

        # At the end, what are we still missing?
        is_missing = ~X.index.isin(X_out.index)
        n_uncorrelated = is_missing.sum()
        w_message = (
            "Did not successfully correlate all filepaths. "
            f"n_uncorrelated = {n_uncorrelated}"
        )
        if n_uncorrelated > 0:
            if self.unhandled_files == "error":
                assert False, w_message
            elif "drop" in self.unhandled_files:
                if "warn" in self.unhandled_files:
                    warnings.warn(w_message)
            elif "passthrough" in self.unhandled_files:
                if "warn" in self.unhandled_files:
                    warnings.warn(w_message)
                X_missing = X.loc[is_missing].copy()
                X_missing["selected"] = False
                X_out["selected"] = True
                X_out = pd.concat([X_out, X_missing])
            else:
                raise ValueError("Unrecognized method for unhandled files.")

        # Organize the index
        # Don't try to sort by indices X_out does not have
        sort_inds = X.index[X.index.isin(X_out.index)]
        X_out = X_out.loc[sort_inds]

        # Select only the desired columns
        if self.output_columns is not None:
            X_out = X_out[self.output_columns]

        return X_out

    def get_logs(self) -> pd.DataFrame:
        """Combine the different logs

        Returns
        -------
        log_df : pd.DataFrame
            Combined dataframe containing IMG, IMU, and GPS metadata for each image.
        """

        img_log_df = self.load_img_log()
        imu_log_df = self.load_imu_log()
        gps_log_df = self.load_gps_log()

        dfs_interped = [
            img_log_df,
        ]
        source_log_names = ["imu", "gps"]
        for i, df_to_include in enumerate([imu_log_df, gps_log_df]):

            source_log_name = source_log_names[i]
            df_to_include = df_to_include.copy()

            # This doesn't interpolate well unless converted
            if "GPSTime" in df_to_include.columns:
                del df_to_include["GPSTime"]

            # Get the timestamps in the right time zone
            df_to_include["CurrTimestamp_in_img_tz"] = df_to_include[
                "CurrTimestamp"
            ] - pd.Timedelta(self.tz_offset_in_hr, "hr")
            df_to_include = (
                df_to_include.dropna(subset=["CurrTimestamp_in_img_tz"])
                .set_index("CurrTimestamp_in_img_tz")
                .sort_index()
            )
            df_to_include["timestamp_int_{}".format(source_log_name)] = df_to_include[
                "CurrTimestamp"
            ].astype(int)
            del df_to_include["CurrTimestamp"]

            # Interpolate
            interp_fn = scipy.interpolate.interp1d(
                df_to_include.index.astype(int), df_to_include.values.transpose()
            )
            interped = interp_fn(img_log_df["timestamp"].astype(int))
            df_interped = pd.DataFrame(
                interped.transpose(), columns=df_to_include.columns
            )

            dfs_interped.append(df_interped)

        log_df = pd.concat(
            dfs_interped,
            axis="columns",
        )

        return log_df

    def load_img_log(self, img_log_fp: str = None) -> pd.DataFrame:
        """Load the images log.

        Parameters
        ----------
        img_log_fp: Location of the image log.
            Defaults to the one provided at init.

        Returns
        -------
        img_log_df: pd.DataFrame
            Dataframe containing the image log.
        """

        assert ("img_log" in self.io_manager.input_filepaths) | (
            "images" in self.io_manager.input_filepaths
        ), "Must provide an input description for either img_log or images."

        try:
            if img_log_fp is None:
                img_log_fp = self.io_manager.input_filepaths["img_log"]

            # Load data
            # Column names are known and input ad below.
            img_log_df = pd.read_csv(
                img_log_fp,
                names=[
                    "odroid_timestamp",
                    "obc_timestamp",
                    "camera_num",
                    "serial_num",
                    "exposure_time",
                    "sequence_ind",
                    "internal_temp",
                    "filename",
                ]
                + [f"Unnamed: {i + 1}" for i in range(12)],
            )
        except (KeyError, IOError) as _:
            # If there's no explicit image log we fall back to the filenames
            img_log_df = utils.check_filepaths_input(
                self.io_manager.input_filepaths["images"]
            )

            # Parse the filepath to make the data frame
            img_log_df["filename"] = img_log_df["filepath"].apply(os.path.basename)
            str_to_parse = img_log_df["filename"].str.split("_img").str[0]
            df_data = list(str_to_parse.str.split("_"))
            img_log_df_addendum = pd.DataFrame(
                df_data,
                columns=[
                    "image_cycle",
                    "timestamp_id",
                    "serial_num",
                    "camera_num",
                ],
            )
            img_log_df = pd.concat(
                [img_log_df, img_log_df_addendum],
                axis="columns",
            )

            # Get the timestamp
            img_log_df["timestamp"] = pd.to_datetime(
                img_log_df["timestamp_id"].astype(float),
                unit="s",
            )

            return img_log_df

        # Parse the timestamp
        # We use a combination of the odroid timestamp and the obc
        # timestamp because the odroid timestamp is missing the year but
        # the obc_timestamp has the wrong month.
        timestamp_split = img_log_df["obc_timestamp"].str.split("_")
        img_log_df["obc_timestamp"] = pd.to_datetime(
            timestamp_split.apply(lambda x: "_".join(x[:2])), format=" %Y%m%d_%H%M%S"
        )
        img_log_df["timestamp"] = pd.to_datetime(
            img_log_df["obc_timestamp"].dt.year.astype(str)
            + " "
            + img_log_df["odroid_timestamp"]
        )
        img_log_df["timestamp_id"] = timestamp_split.apply(lambda x: x[-1]).astype(int)
        # Correct for overflow
        img_log_df.loc[img_log_df["timestamp_id"] < 0, "timestamp_id"] += 2**32
        img_log_df["timestamp_id"] = img_log_df["timestamp_id"].astype(str)

        # Drop unnamed columns
        img_log_df = img_log_df.drop(
            [column for column in img_log_df.columns if "Unnamed" in column],
            axis="columns",
        )

        # Get filepaths
        img_log_df["obc_filename"] = img_log_df["filename"].copy()
        img_log_df["filename"] = img_log_df["obc_filename"].apply(os.path.basename)

        return img_log_df

    def load_imu_log(self, imu_log_fp: str = None) -> pd.DataFrame:
        """Load the IMU log.

        Parameters
        ----------
        imu_log_fp: Location of the IMU log.
            Defaults to the one provided at init.

        Returns
        -------
        imu_log_df: pd.DataFrame
            Dataframe containing the IMU log.
        """

        if imu_log_fp is None:
            imu_log_fp = self.io_manager.input_filepaths["imu_log"]

        imu_log_df = pd.read_csv(imu_log_fp, low_memory=False)

        # Remove the extra header rows, and the nan rows
        imu_log_df.dropna(
            subset=[
                "CurrTimestamp",
            ],
            inplace=True,
        )
        imu_log_df.drop(
            imu_log_df.index[imu_log_df["CurrTimestamp"] == "CurrTimestamp"],
            inplace=True,
        )

        # Handle some situations where the pressure is negative
        ac_columns = ["TempC", "pressure", "mAltitude"]
        imu_log_df.loc[imu_log_df["pressure"].astype(float) < 0] = np.nan

        # Convert to datetime and sort
        imu_log_df["CurrTimestamp"] = pd.to_datetime(imu_log_df["CurrTimestamp"])
        imu_log_df.sort_values("CurrTimestamp", inplace=True)

        # Assign dtypes
        skipped_cols = []
        for column in imu_log_df.columns:
            if column == "CurrTimestamp":
                continue

            imu_log_df[column] = imu_log_df[column].astype(float)

        # Now also handle when the altitude or temperature are weird
        imu_log_df.loc[imu_log_df["TempC"] < -273, ac_columns] = np.nan
        imu_log_df.loc[imu_log_df["mAltitude"] < 0.0, ac_columns] = np.nan
        imu_log_df.loc[imu_log_df["mAltitude"] > 20000.0, ac_columns] = np.nan

        # Get gyro magnitude
        imu_log_df["imuGyroMag"] = np.linalg.norm(
            imu_log_df[["imuGyroX", "imuGyroY", "imuGyroZ"]], axis=1
        )

        return imu_log_df

    def load_gps_log(
        self, gps_log_fp: str = None, latlon_crs: str = "EPSG:4326"
    ) -> pd.DataFrame:
        """Load the GPS log.

        Parameters
        ----------
        gps_log_fp:
            Location of the GPS log.
            Defaults to the one provided at init.
        latlon_crs: str
            The CRS of the latitude and longitude coordinates.

        Returns
        -------
        gps_log_df: pd.DataFrame
            Dataframe containing the GPS log.
        """

        if gps_log_fp is None:
            gps_log_fp = self.io_manager.input_filepaths["gps_log"]

        gps_log_df = pd.read_csv(gps_log_fp)

        # Remove the extra header rows and the empty rows
        gps_log_df.dropna(
            subset=[
                "CurrTimestamp",
            ],
            inplace=True,
        )
        gps_log_df.drop(
            gps_log_df.index[gps_log_df["CurrTimestamp"] == "CurrTimestamp"],
            inplace=True,
        )

        # Remove the empty rows
        empty_timestamp = "00.00.0000 00:00:00000"
        gps_log_df.drop(
            gps_log_df.index[gps_log_df["CurrTimestamp"] == empty_timestamp],
            inplace=True,
        )

        # Convert to datetime and sort
        gps_log_df["CurrTimestamp"] = pd.to_datetime(gps_log_df["CurrTimestamp"])
        gps_log_df.sort_values("CurrTimestamp", inplace=True)

        # Assign dtypes
        for column in gps_log_df.columns:
            if column in ["CurrTimestamp", "GPSTime"]:
                continue

            gps_log_df[column] = gps_log_df[column].astype(float)

        # Coordinates
        if isinstance(latlon_crs, str):
            latlon_crs = pyproj.CRS(latlon_crs)
        latlon_crs_to_crs = pyproj.Transformer.from_crs(latlon_crs, self.crs)
        gps_log_df["sensor_x"], gps_log_df["sensor_y"] = latlon_crs_to_crs.transform(
            gps_log_df["GPSLat"], gps_log_df["GPSLong"]
        )

        return gps_log_df


class GeoTIFFPreprocessor(TransformerMixin, BaseEstimator):
    """Class for getting geotransforms out from a list of GeoTIFF filepaths."""

    def __init__(
        self,
        crs: pyproj.CRS,
        spatial_error: float = 0.0,
        padding_fraction: float = 0.1,
        passthrough: bool = True,
    ):
        """
        Initialize the Preprocessor object.

        Parameters
        ----------
        crs : pyproj.CRS
            The coordinate reference system (CRS) to use for the data.
            It can be specified as a string in the format 'EPSG:<code>' or as a
            pyproj.CRS object. The default is 'EPSG:3857'.
        passthrough : bool, optional
            Flag indicating input columns that can passthrough without any
            preprocessing. If True, all unhandled columns will be passed through.
            The default is True.
        spatial_error : float, optional
            The spatial error in meters. This parameter provides a base error
            which can be used in error propagation.
        padding_fraction : float, optional
            The fraction of padding to add around the data extent in units
            of image hypotenuse. Default is 0.1
        """
        self.crs = crs
        self.required_columns = ["filepath"]
        self.spatial_error = spatial_error
        self.padding_fraction = padding_fraction
        self.passthrough = passthrough

    @utils.enable_passthrough
    def fit(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ) -> "GeoTIFFPreprocessor":
        """
        Fits the GeoTIFFPreprocessor to the input data.

        Parameters
        ----------
        X : Union[np.ndarray[str], list[str], pd.DataFrame]
            The input filepaths to fit the preprocessor on.

        y : None, optional
            The target variable (default is None).

        Returns
        -------
        GeoTIFFPreprocessor
            The fitted GeoTIFFPreprocessor object.
        """

        X = utils.check_filepaths_input(X, required_columns=self.required_columns)

        self.is_fitted_ = True
        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: Union[np.ndarray[str], list[str], pd.DataFrame],
        y=None,
    ) -> pd.DataFrame:
        """Transform the input dataframe to also hold geotransform information.

        Parameters
        ----------
        X : Union[np.ndarray[str], list[str], pd.DataFrame]
            The input filepaths to be transformed.
        y : None, optional
            The target variable, if applicable. Default is None.

        Returns
        -------
        pd.DataFrame
            The transformed data as a DataFrame.
        """

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Loop over and get datasets
        rows = []
        for i, fp in enumerate(tqdm.tqdm(X["filepath"], ncols=80)):

            # Try to load the dataset
            try:
                dataset = gdal.Open(fp, gdal.GA_ReadOnly)
            except Exception as e:
                # We have to be kind of obtuse about catching this error
                # because gdal doesn't have nice error classes
                if gdal.GetLastErrorType() != gdal.CPLE_FileIO:
                    raise e
                dataset = None
            if dataset is None:
                row = pd.Series(
                    [
                        np.nan,
                    ]
                    * len(GEOTRANSFORM_COLS),
                    index=GEOTRANSFORM_COLS,
                    name=X.index[i],
                )
                rows.append(row)
                continue

            x_min, pixel_width, x_rot, y_max, y_rot, pixel_height = (
                dataset.GetGeoTransform()
            )

            # Get bounds
            x_max = x_min + pixel_width * dataset.RasterXSize
            y_min = y_max + pixel_height * dataset.RasterYSize

            # Convert to desired crs.
            dataset_crs = pyproj.CRS(dataset.GetProjection())
            dataset_crs_to_crs = pyproj.Transformer.from_crs(
                dataset_crs, self.crs, always_xy=True
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

            spatial_error = self.spatial_error
            hypotenuse = np.sqrt((x_max - x_min) ** 2.0 + (y_max - y_min) ** 2.0)
            padding = self.padding_fraction * hypotenuse

            row = pd.Series(
                [
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    pixel_width,
                    pixel_height,
                    x_rot,
                    y_rot,
                    dataset.RasterXSize,
                    dataset.RasterYSize,
                    x_center,
                    y_center,
                    spatial_error,
                    padding,
                ],
                index=GEOTRANSFORM_COLS,
                name=X.index[i],
            )
            rows.append(row)

        new_df = pd.DataFrame(rows)

        X = pd.concat([X, new_df], axis="columns")

        return X
