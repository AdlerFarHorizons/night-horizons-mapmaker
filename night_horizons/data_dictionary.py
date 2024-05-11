"""This module contains descriptions for common data products.
"""

import numpy as np
import pandas as pd

DATA_DICTONARIES = {
    "example_data_source": {
        "example_var": {
            "description": "This is an example variable that demonstrates formats.",
            "type": str,
            "units": np.nan,
        },
    },
    "metadata": {
        "ClimbErr": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "GPSAlt": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "GPSHeading": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "LatErr": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "LonErr": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "SpeerErr": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "TempC": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "camera_num": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "center_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "d_to_center": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "duration": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "exposure_time": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "filename": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "filename_x": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "filename_y": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "filepath": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "imuAccelX": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuAccelY": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuAccelZ": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuGyroMag": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuGyroX": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuGyroY": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuGyroZ": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuMagX": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuMagY": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuMagZ": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuPitch": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuRoll": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "imuYaw": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "internal_temp": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "mAltitude": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "obc_filename": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "obc_timestamp": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "odroid_timestamp": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "order": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "output_filepath": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "padding": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "pixel_height": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "pixel_height_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "pixel_width": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "pixel_width_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "pressure": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "return_code": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "score": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "sensor_order": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "sensor_x": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "sensor_y": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "sequence_ind": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "serial_num": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "spatial_error": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "timestamp": {
            "description": "",
            "type": np.object,
            "units": np.nan,
        },
        "timestamp_id": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "timestamp_int_gps": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "timestamp_int_imu": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_center": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_max": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_max_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_min": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_min_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_off": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "x_rot": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "x_size": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "x_size_diff": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "y_center": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_max": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_max_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_min": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_min_diff": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_off": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "y_rot": {
            "description": "",
            "type": np.float64,
            "units": np.nan,
        },
        "y_size": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
        "y_size_diff": {
            "description": "",
            "type": np.int64,
            "units": np.nan,
        },
    },
}


def get_data_dictionary(data_source: str, data: pd.DataFrame = None) -> dict[dict]:
    """Get a dictionary that describes the columns in a data source.

    Parameters
    ----------
    data_source : str
        The name of the data source. This should be one of the keys in DATA_DICTONARIES.

    data : pd.DataFrame, optional
        The data to retrieve the column names for. If not provided, the column names
        will be retrieved from the full data dictionary.

    Returns
    -------
    dict[dict]
        A dictionary that describes the columns in the data source. Each column has
        a dictionary containing the description, data type, and units.
    """

    # Get the data dictionary
    full_data_dict = DATA_DICTONARIES[data_source]

    if data is None:
        columns = full_data_dict.keys()
    else:
        columns = data.columns

    data_dict = {}
    for column in columns:
        data_dict[column] = full_data_dict[column]

    return full_data_dict
