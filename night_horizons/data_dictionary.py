"""This module contains descriptions for common data products.
"""

import numpy as np
import pandas as pd

DATA_DICTONARIES = {
    "example_data_source": {
        "example_var": {
            "description": "This is an example variable that demonstrates formats.",
            "type": str,
            "units": "",
        },
    },
    "metadata": {
        "ClimbErr": {
            "description": "TBD",
            "type": np.float64,
            "units": "TBD",
        },
        "GPSAlt": {
            "description": "Altitude according to the GPS.",
            "type": np.float64,
            "units": "TBD",
        },
        "GPSHeading": {
            "description": "TBD",
            "type": np.float64,
            "units": "TBD",
        },
        "LatErr": {
            "description": "TBD",
            "type": np.float64,
            "units": "TBD",
        },
        "LonErr": {
            "description": "TBD",
            "type": np.float64,
            "units": "TBD",
        },
        "SpeerErr": {
            "description": "TBD",
            "type": np.float64,
            "units": "TBD",
        },
        "TempC": {
            "description": "Temperature.",
            "type": np.float64,
            "units": "Celcius",
        },
        "camera_num": {
            "description": (
                "Camera number, i.e. 0, 1, or 2 for the side, nadir, and "
                "the other side respectively."
            ),
            "type": np.int64,
            "units": "",
        },
        "center_diff": {
            "description": (
                "Difference between true image center and estimated image center, "
                "if known."
            ),
            "type": np.float64,
            "units": "meters",
        },
        "distance_to_starting_image": {
            "description": (
                "Distance to the center of the mosaic, where the center of the mosaic "
                "is defined by transformers.order.SensorAndDistanceOrder as the "
                "first fit image."
            ),
            "type": np.float64,
            "units": "meters",
        },
        "duration": {
            "description": (
                "How long it took to perform the selected operation on the image."
            ),
            "type": np.float64,
            "units": "seconds",
        },
        "exposure_time": {
            "description": "How long the camera was exposed to light.",
            "type": np.int64,
            "units": "seconds",
        },
        "filename": {
            "description": "Name of the file within the analysis environment.",
            "type": str,
            "units": "",
        },
        "filename_x": {
            "description": "Alternative filename.",
            "type": str,
            "units": "",
        },
        "filename_y": {
            "description": "Alternative filename.",
            "type": str,
            "units": "",
        },
        "filepath": {
            "description": "Location of the file within the analysis environment.",
            "type": str,
            "units": "",
        },
        "imuAccelX": {
            "description": "X-direction acceleration measured by the IMU.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuAccelY": {
            "description": "Y-direction acceleration measured by the IMU.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuAccelZ": {
            "description": "Z-direction acceleration measured by the IMU.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuGyroMag": {
            "description": "Magnitude of the IMU gyroscope value.",
            "type": np.float64,
            "units": "",
        },
        "imuGyroX": {
            "description": "X-direction IMU gyroscope value.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuGyroY": {
            "description": "Y-direction IMU gyroscope value.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuGyroZ": {
            "description": "Z-direction IMU gyroscope value.",
            "type": np.float64,
            "units": "TBD",
        },
        "imuMagX": {
            "description": "X-direction IMU magnetometer value.",
            "type": np.float64,
            "units": "",
        },
        "imuMagY": {
            "description": "Y-direction IMU magnetometer value.",
            "type": np.float64,
            "units": "",
        },
        "imuMagZ": {
            "description": "Z-direction IMU magnetometer value.",
            "type": np.float64,
            "units": "",
        },
        "imuPitch": {
            "description": (
                "Pitch, estimated from the IMU instrument measurements. "
                "As of writing, not robust."
            ),
            "type": np.float64,
            "units": "",
        },
        "imuRoll": {
            "description": (
                "Roll, estimated from the IMU instrument measurements. "
                "As of writing, not robust."
            ),
            "type": np.float64,
            "units": "",
        },
        "imuYaw": {
            "description": (
                "Yaw, estimated from the IMU instrument measurements. "
                "As of writing, not robust."
            ),
            "type": np.float64,
            "units": "",
        },
        "internal_temp": {
            "description": "TBD",
            "type": np.int64,
            "units": "TBD",
        },
        "mAltitude": {
            "description": "Altitude of the baloon, according to the ACS.",
            "type": np.float64,
            "units": "meters",
        },
        "obc_filename": {
            "description": "Name of the image, according to the onboard computer.",
            "type": str,
            "units": "",
        },
        "obc_timestamp": {
            "description": "Timestamp of the image, according to the onboard computer.",
            "type": str,
            "units": "",
        },
        "odroid_timestamp": {
            "description": "TBD",
            "type": str,
            "units": "TBD",
        },
        "order": {
            "description": "Order in which to analyze the images.",
            "type": np.int64,
            "units": "",
        },
        "output_filepath": {
            "description": (
                "Location the output image (usually a referenced image) is saved at."
            ),
            "type": str,
            "units": "",
        },
        "padding": {
            "description": (
                "When performing image registration, the search zone is of a size "
                "(width + 2 * padding, height + 2 * padding.)"
            ),
            "type": np.float64,
            "units": "meters",
        },
        "pixel_height": {
            "description": "Height of an individual pixel.",
            "type": np.float64,
            "units": "meters",
        },
        "pixel_height_diff": {
            "description": (
                "Difference between the estimated and actual height of a "
                "pixel. Used when testing."
            ),
            "type": np.float64,
            "units": "meters",
        },
        "pixel_width": {
            "description": "Width of an individual pixel."
            "type": np.float64,
            "units": "meters",
        },
        "pixel_width_diff": {
            "description": (
                "Difference between the estimated and actual width of a "
                "pixel. Used when testing."
            ),
            "type": np.float64,
            "units": "meters",
        },
        "pressure": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "return_code": {
            "description": "",
            "type": np.object,
            "units": "",
        },
        "score": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "sensor_order": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "sensor_x": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "sensor_y": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "sequence_ind": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "serial_num": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "spatial_error": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "timestamp": {
            "description": "",
            "type": np.object,
            "units": "",
        },
        "timestamp_id": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "timestamp_int_gps": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "timestamp_int_imu": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_center": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_max": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_max_diff": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_min": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_min_diff": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_off": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "x_rot": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "x_size": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "x_size_diff": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "y_center": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_max": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_max_diff": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_min": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_min_diff": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_off": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "y_rot": {
            "description": "",
            "type": np.float64,
            "units": "",
        },
        "y_size": {
            "description": "",
            "type": np.int64,
            "units": "",
        },
        "y_size_diff": {
            "description": "",
            "type": np.int64,
            "units": "",
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
