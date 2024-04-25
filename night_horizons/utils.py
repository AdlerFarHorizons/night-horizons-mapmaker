from functools import wraps
import glob
import inspect
import os
from typing import Tuple, Union

import cv2
import logging
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array


def discover_data(
    directory: str,
    extension: Union[str, list[str]] = None,
    pattern: str = None,
) -> pd.Series:
    '''
    TODO: Delete?

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
        glob_pattern = os.path.join(directory, '**', '*.*')
        fps = glob.glob(glob_pattern, recursive=True)
    # When a single extension
    elif isinstance(extension, str):
        glob_pattern = os.path.join(directory, '**', f'*{extension}')
        fps = glob.glob(glob_pattern, recursive=True)
    # When a list of extensions
    else:
        try:
            fps = []
            for ext in extension:
                glob_pattern = os.path.join(directory, '**', f'*{ext}')
                fps.extend(glob.glob(glob_pattern, recursive=True))
        except TypeError:
            raise TypeError(f'Unexpected type for extension: {extension}')

    fps = pd.Series(fps)

    # Filter to select particular files
    if pattern is not None:
        contains_pattern = fps.str.findall(pattern).str[0].notna()
        fps = fps.loc[contains_pattern]

    fps.index = np.arange(fps.size)

    return fps


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
    elif isinstance(X, pd.Series):
        X = pd.DataFrame(X, columns=['filepath'])
        return X

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

    return X.copy()


def check_columns(
    actual: Union[pd.Series, list[str]],
    expected: Union[pd.Series, list[str]],
    passthrough: Union[bool, list[str]] = False
):
    '''Check that the columns of a dataframe are as required.

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

    expected = pd.Series(expected)
    required_not_in_actual = ~expected.isin(actual)
    assert required_not_in_actual.sum() == 0, (
        f'Missing columns {expected.loc[required_not_in_actual]}'
    )

    if isinstance(passthrough, bool):
        assert passthrough or (len(actual) == len(expected)), (
            f'Expected columns {list(expected)}.\n'
            f'Got columns {list(actual)}.'
        )
    else:
        expected = pd.unique(pd.Series(list(expected) + list(passthrough)))
        assert len(expected) == len(actual), (
            f'Expected columns {expected}.\n'
            f'Got columns {list(actual)}.'
        )

    return actual


def enable_passthrough(func):
    '''
    Maybe deprecate this in the future...
    I forgot that the possibly better method is to make use of
    sklearn.compose.ColumnTransformer.
    There's also relevant functionality built into sklearn.pipeline.

    Columns that are neither in self.passthrough or self.required_columns
    are dropped.

    Consider a column `A` that is in self.passthrough,
    an input dataframe X, and an output dataframe X_out.

    Rules:
    1. If `A` is in X.columns and is not in X_out.columns,
        include the original `A` in X_out.
    2. If `A` is in X.columns and is also in X_out.columns,
        keep `A` from X_out unaltered.
    3. If `A` is not in X.columns, ignore it.

    Parameters
    ----------
    Returns
    -------
    Requires
    --------
    self.passthrough : Union[list[str], bool]

    '''

    def wrapper(
        self,
        X: Union[pd.DataFrame, pd.Series],
        *args,
        **kwargs
    ):

        if isinstance(X, pd.Series):
            return func(self, X, *args, **kwargs)

        # Boolean passthrough values
        if isinstance(self.passthrough, bool):
            if self.passthrough:
                passthrough = X.columns
            else:
                check_columns(X.columns, self.required_columns)
                passthrough = pd.Series([])

        # When specific columns are passed
        else:
            passthrough = pd.Series(self.passthrough)
            is_in_X = passthrough.isin(X.columns)
            passthrough = passthrough[is_in_X]

        # Select the columns to pass in
        X_in = X[self.required_columns].copy()

        # Call function
        X_out = func(self, X_in, *args, **kwargs)

        # When there's nothing to passthrough
        if len(passthrough) == 0:
            return X_out

        # Don't try to add columns to something that is not a dataframe
        if not isinstance(X_out, pd.DataFrame):
            return X_out

        # Re-add passthrough columns
        is_in_X_out = passthrough.isin(X_out.columns)
        passthrough_cols = passthrough[~is_in_X_out]
        X_out = X_out.join(X[passthrough_cols])

        return X_out

    return wrapper


class LoggerMixin:
    '''
    Note that a decorator is not possible because we're typically
    interested in local variables.

    NOTE: It is tempting to refactor this to use dependency injection,
        but in practice that requires one more object to be passed around,
        and for something as pervasive as the logger, that's a lot of
        boilerplate. I think it's better to just use a mixin.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        log_keys: list[str] = [],
    ):
        self.log_keys = log_keys

    def start_logging(self):
        self.log = {}

    @property
    def log(self):
        '''We create the log on the fly so that it's not stored in memory.
        '''
        if not hasattr(self, '_log'):
            self._log = {}
        return self._log

    @log.setter
    def log(self, value):
        self._log = value

    def update_log(self, new_dict: dict, target: dict = None):

        if len(self.log_keys) == 0:
            return target

        if target is None:
            target = self.log

        # Filter out values that aren't being tracked
        new_dict = {
            log_key: item
            for log_key, item in new_dict.items()
            if log_key in self.log_keys
        }
        target.update(new_dict)

        return target

    def stop_logging(self):
        self.log_keys = []


class LoopLoggerMixin(LoggerMixin):

    def start_logging(self, i_start: int = 0, log_filepath: str = None):
        '''
        Parameters
        ----------
        Returns
        -------

        Attributes Modified
        -------------------
        log : dict
            Dictionary for variables the user may want to view. This should
            be treated as "read-only".

        logs : list[dict]
            List of dictionaries for variables the user may want to view.
            One dictionary per image.
            Each should be treated as "read-only".
        '''

        # It's harder to create the log via properties when possibly loading
        # from a file.
        self.log = {}
        self.logs = []

        # Open the log if available
        if isinstance(log_filepath, str) and os.path.isfile(log_filepath):
            log_df = pd.read_csv(log_filepath, index_col=0)

            # Format the stored logs
            for i, ind in enumerate(log_df.index):
                if i >= i_start:
                    break
                log = dict(log_df.loc[ind])
                self.logs.append(log)

    def write_log(self, log_filepath: str):

        log_df = pd.DataFrame(self.logs)
        log_df.to_csv(log_filepath)


class StdoutLogger(object):
    def __init__(self, logger, stdout):
        self.logger = logger
        self.stdout = stdout

    def write(self, message):
        if message.strip() != "":
            self.logger.info('stdout: ' + message.strip())
        self.stdout.write(message)

    def flush(self):
        self.stdout.flush()


def get_logger(name: str = None):

    # Get logger
    logger = logging.getLogger(name)

    # Get logging level
    # Options are DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_level = os.getenv('LOGGING_LEVEL')
    if logging_level is None:
        logging_level = 'WARNING'

    # Convert to numeric value and set logging level
    logging_level = getattr(logging, logging_level)
    logger.setLevel(logging_level)

    return logger


def deep_merge(orig_dict, new_dict):
    result = orig_dict.copy()
    for key, value in new_dict.items():
        if (
            isinstance(value, dict)
            and (key in orig_dict)
            and isinstance(orig_dict[key], dict)
        ):
            result[key] = deep_merge(orig_dict[key], value)
        else:
            result[key] = value

    return result
