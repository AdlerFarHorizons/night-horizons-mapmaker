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
