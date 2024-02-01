'''
TODO: Refactor into classes with @staticmethod. Will be more clean.
'''

from functools import wraps
import glob
import inspect
import os
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
import pyproj
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


def load_image(
    filepath: str,
    dtype: type = np.uint8,
    img_shape: Tuple = (1200, 1920),
):
    '''Load an image from disk.
    TODO: Delete?

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


def get_distance(crs, x1, y1, x2, y2):

    if crs.is_geographic:

        geod = crs.get_geod()
        _, _, distance = geod.inv(
            lons1=x1, lats1=y1,
            lons2=x2, lats2=y2,
        )
    else:
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance


def update_row(df, row):
    '''
    Incorporate the row into the output DataFrame
    We drop and append because concat handles adding new columns,
    while Z_out.loc[ind] = row does not.
    As set up, this is probably pretty slow.

    Parameters
    ----------
    Returns
    -------
    '''

    # Check index, and preserve
    assert row.name in df.index, 'Row name not in index.'
    original_index = df.index

    # Drop so that we avoid duplicates
    df = df.drop(row.name)

    if len(df) > 0:
        df = pd.concat([df, row.to_frame().T])
    else:
        df = row.to_frame().T

    return df.loc[original_index]


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


def store_parameters(constructor):
    '''Decorator for automatically storing arguments passed to a constructor.
    I.e. any args passed to constructor via
    my_object = MyClass(*args, **kwargs)
    will be stored in my_object as an attribute, i.e. my_object.arg

    TODO: Deprecate this, and consider storing the many parameters
    in a different, more-readable way.
    Probably as options dictionaries.
    This would include maintaining consistent rules for creating objects
    vs passing them in.
    Also consistent rules for using super().__init__ to store options
    vs storing them directly. I'm leaning towards requiring super().__init__
    because then the user can track what parameters exist in the superclass
    vs subclass.
    Think also about how fit parameters are handled.
    Also, consistent names for image_operator vs image_blender.

    Parameters
    ----------
        constructor : callable
            Constructor to wrap.
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


class ReferencedRawSplit:

    def __init__(
        self,
        io_manager,
        test_size: Union[int, float] = 0.2,
        random_state: Union[int, np.random.RandomState] = None,
    ):

        self.io_manager = io_manager
        self.test_size = test_size
        self.random_state = random_state

    def train_test_production_split(
        self
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        '''TODO: This is an awkward spot for this function.

        Parameters
        ----------
        Returns
        -------
        '''

        referenced_fps = self.io_manager.input_filepaths['referenced_images']
        raw_fps = self.io_manager.input_filepaths['raw_images']

        fps_train, fps_test = train_test_split(
            referenced_fps,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Combine raw fps and test fps
        raw_fps.index += referenced_fps.size
        fps = pd.concat([fps_test, raw_fps])

        fps = fps.sample(
            n=fps.size,
            replace=False,
            random_state=self.random_state,
        )

        return fps_train, fps_test, fps


class LoggerMixin:
    '''
    Note that a decorator is not possible because we're typically
    interested in local variables.

    TODO: Evaluate if this helps things or makes things worse.
    When debugging I wanted to be able to peek at any of the parameters,
    but in a statistical approach. The idea was that this would be helpful
    for identify blackbox parameters that correlated with successful
    (or poor) results.

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
