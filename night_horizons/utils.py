from typing import Union

import numpy as np
import pandas as pd
import scipy
from sklearn.utils.validation import check_array
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.


def check_filepaths_input(
    X: Union[np.ndarray[str], list[str], pd.DataFrame],
    required_columns: list[str] = ['filepath'],
    passthrough: bool = False,
) -> pd.DataFrame:
    '''We need a list of the filepaths or a dataframe with one of the column
    being the filepaths.

    Parameters
    ----------
        X
            Input data.
        required_columns
            The columns required if a dataframe is passed.
        passthrough
            If True, allow columns other than the required columns.

    Returns
    -------
        X
            Checked input data, possibly reshaped.
    '''

    if isinstance(X, pd.DataFrame):
        return check_df_input(X, required_columns, passthrough)

    # We offer some minor reshaping to be compatible with common
    # expectations that a single list of features doesn't need to be 2D.
    if len(np.array(X).shape) == 1:
        X = np.array(X).reshape(1, len(X))

    # Check and unpack X
    X = check_array(X, dtype='str')
    X = pd.DataFrame(X.transpose(), columns=['filepath'])

    return X


def check_df_input(
    X: Union[np.ndarray[str], list[str], pd.DataFrame],
    required_columns: list[str],
    passthrough: bool = False
):
    '''Check that we have a dataframe with the right columns.

    Parameters
    ----------
        X
            Input data.
        required_columns
            The columns required if a dataframe is passed.
        passthrough
            If True, allow columns other than the required columns.

    Returns
    -------
        X
            Validated input data.
    '''

    assert isinstance(X, pd.DataFrame), 'Expected a pd.DataFrame.'

    check_columns(X.columns, required_columns, passthrough)

    return X


def check_columns(
    actual: pd.Series,
    required: list[str],
    passthrough: bool = False
):
    '''
    Parameters
    ----------
        actual
            Actual columns.
        required
            The columns required.
        passthrough
            If True, allow columns other than the required columns.

    Returns
    -------
        actual
            The validated columns.
    '''

    required = pd.Series(required)
    required_not_in_actual = ~required.isin(actual)
    assert required_not_in_actual.sum() == 0, (
        f'Missing columns {required.loc[required_not_in_actual]}'
    )

    assert passthrough or len(actual) == len(required), (
        f'Expecting *only* columns {required}.'
    )

    return actual
