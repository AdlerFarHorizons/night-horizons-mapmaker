"""This module contains descriptions for common data products.
"""

import numpy as np
import pandas as pd

DATA_DICTIONARY = {
    "example_var": {
        "description": "This is an example variable that demonstrates formats.",
        "type": str,
        "units": "",
    },
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
        "description": "Width of an individual pixel.",
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
        "description": "Pressure measured by the ACS.",
        "type": np.float64,
        "units": "TBD",
    },
    "return_code": {
        "description": "Whether the operation was successful, and if not, why.",
        "type": str,
        "units": "",
    },
    "score": {
        "description": "Score quantifying accuracy, if available.",
        "type": np.float64,
        "units": "",
    },
    "sensor_order": {
        "description": "Order in which the images are to be referenced, as determined by the sensors.",
        "type": np.int64,
        "units": "",
    },
    "sensor_x": {
        "description": "Location of the sensor (i.e. the balloon) in the x-direction.",
        "type": np.float64,
        "units": "meters",
    },
    "sensor_y": {
        "description": "Location of the sensor (i.e. the balloon) in the y-direction.",
        "type": np.float64,
        "units": "meters",
    },
    "sequence_ind": {
        "description": "Which image in a given sequence this is.",
        "type": np.int64,
        "units": "",
    },
    "serial_num": {
        "description": "Serial number of the camera that took the image.",
        "type": np.int64,
        "units": "",
    },
    "spatial_error": {
        "description": "Approximate error in the xy dimensions of the image.",
        "type": np.float64,
        "units": "",
    },
    "timestamp": {
        "description": "Time at which the image was taken.",
        "type": str,
        "units": "",
    },
    "timestamp_id": {
        "description": "Unique ID constructed from the timestamp.",
        "type": np.int64,
        "units": "",
    },
    "timestamp_int_gps": {
        "description": "Timestamp for the GPS log entry, as an integer (epoch time).",
        "type": np.float64,
        "units": "",
    },
    "timestamp_int_imu": {
        "description": "Timestamp for the IMU log entry, as an integer (epoch time).",
        "type": np.float64,
        "units": "",
    },
    "x_center": {
        "description": "Estimated center of the image in the x-direction.",
        "type": np.float64,
        "units": "meters",
    },
    "x_max": {
        "description": "Estimated maximum x-value of the image.",
        "type": np.float64,
        "units": "meters",
    },
    "x_max_diff": {
        "description": "Difference between the estimated and actual maximum x-value of the image, if available.",
        "type": np.float64,
        "units": "meters",
    },
    "x_min": {
        "description": "Estimated minimum x-value of the image.",
        "type": np.float64,
        "units": "meters",
    },
    "x_min_diff": {
        "description": "Difference between the estimated and actual minimum x-value of the image, if available.",
        "type": np.float64,
        "units": "meters",
    },
    "x_off": {
        "description": "Pixel offset in the x-direction, used as part of the geo transform.",
        "type": np.int64,
        "units": "",
    },
    "x_rot": {
        "description": "Pixel rotation in the x-direction, used as part of the geo transform. Typically 0.",
        "type": np.float64,
        "units": "",
    },
    "x_size": {
        "description": "Length of the image in pixels in the x direction, used as part of the geo transform.",
        "type": np.int64,
        "units": "",
    },
    "x_size_diff": {
        "description": "Difference between the estimated x size in the pixel frame, and the actual size.",
        "type": np.int64,
        "units": "",
    },
    "y_center": {
        "description": "Estimated center of the image in the y-direction.",
        "type": np.float64,
        "units": "meters",
    },
    "y_max": {
        "description": "Estimated maximum y-value of the image.",
        "type": np.float64,
        "units": "meters",
    },
    "y_max_diff": {
        "description": "Difference between the estimated and actual maximum y-value of the image, if available.",
        "type": np.float64,
        "units": "meters",
    },
    "y_min": {
        "description": "Estimated minimum y-value of the image.",
        "type": np.float64,
        "units": "meters",
    },
    "y_min_diff": {
        "description": "Difference between the estimated and actual minimum y-value of the image, if available.",
        "type": np.float64,
        "units": "meters",
    },
    "y_off": {
        "description": "Pixel offset in the y-direction, used as part of the geo transform.",
        "type": np.int64,
        "units": "",
    },
    "y_rot": {
        "description": "Pixel rotation in the y-direction, used as part of the geo transform. Typically 0.",
        "type": np.float64,
        "units": "",
    },
    "y_size": {
        "description": "Length of the image in pixels in the y direction, used as part of the geo transform.",
        "type": np.int64,
        "units": "",
    },
    "y_size_diff": {
        "description": "Difference between the estimated y size in the pixel frame, and the actual size.",
        "type": np.int64,
        "units": "",
    },
}


def get_data_dictionary(fields: list = None) -> dict[dict]:
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

    # Get the columns to use
    if fields is None:
        fields = DATA_DICTIONARY.keys()

    # Fill in the dictionary
    data_dict = {}
    for field in fields:
        try:
            data_dict[field] = DATA_DICTIONARY[field]
        except KeyError:
            data_dict[field] = {
                "description": "This field was not found in the data dictionary.",
                "type": "Unknown",
                "units": "Unknown",
            }

    return data_dict
