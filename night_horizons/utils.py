'''
TODO: Refactor into classes with @staticmethod. Will be more clean.
'''
import glob
import inspect
import os
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.validation import check_array
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.


def discover_data(
    directory: str,
    extension: Union[str, list[str]] = None,
    pattern: str = None,
) -> pd.Series:
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


def load_image(
    filepath: str,
    dtype: type = np.uint8,
    img_shape: Tuple = (1200, 1920),
):
    '''Load an image from disk.

    Parameters
    ----------
        filepath
            Location of the image.
        dtype
            Datatype. Defaults to integer from 0 to 255
    Returns
    -------
    '''

    ext = os.path.splitext(filepath)[1]

    # Load and reshape raw image data.
    if ext == '.raw':

        raw_img = np.fromfile(filepath, dtype=np.uint16)
        raw_img = raw_img.reshape(img_shape)

        img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)
        img_max = 2**12 - 1

    elif ext in ['.tiff', '.tif']:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # CV2 defaults to BGR, but RGB is more standard for our purposes
        img = img[:, :, ::-1]
        img_max = np.iinfo(img.dtype).max

    else:
        raise IOError('Cannot read filetype {}'.format(ext))

    if img is None:
        return img

    # When no conversion needs to be done
    if img.dtype == dtype:
        return img

    # Rescale
    img = img / img_max
    img = (img * np.iinfo(dtype).max).astype(dtype)

    return img


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
    TODO: Maybe deprecate this....
    I forgot that the possibly better method is to make use of
    sklearn.compose.ColumnTransformer.
    There's also stuff built into pipeline, I think...

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
    '''

    def wrapper(
        self,
        X: Union[pd.DataFrame, pd.Series],
        *args,
        **kwargs
    ):

        if isinstance(X, pd.Series):
            return func(self, X, *args, **kwargs)

        # Get the columns to pass through
        passthrough = pd.Series(self.passthrough)
        is_in_X = pd.Series(passthrough).isin(X.columns)
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


from functools import wraps


def store_parameters(constructor):
    '''Decorator for automatically storing arguments passed to a constructor.
    I.e. any args passed to constructor via
    test_object = TestObject(*args, **kwargs)
    will be stored in test_object, e.g. test_object.args

    Args:
        constructor (function) : Constructor to wrap.
    '''

    @wraps(constructor)
    def wrapped_constructor(self, *args, **kwargs):

        constructor(self, *args, **kwargs)

        parameters_to_store = inspect.getcallargs(
            constructor,
            self,
            *args,
            **kwargs
        )

        for param_key, param_value in parameters_to_store.items():
            if param_key == 'self':
                continue
            setattr(self, param_key, param_value)

    return wrapped_constructor


class LoggerMixin:
    '''
    Note that a decorator is not possible because we're typically
    interested in local variables.

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

    def update_log(self, new_dict: dict, target: dict = None):

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


class LoopLoggerMixin(LoggerMixin):

    def start_logging(self, i_start: int, log_filepath: str):
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

        self.log = {}
        self.logs = []

        # Open the log if available
        if os.path.isfile(log_filepath):
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
