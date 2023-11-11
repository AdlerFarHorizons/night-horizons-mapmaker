import glob
import os
from typing import Union

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


def load_image(filepath: str, dtype: type = np.uint8):
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

    # Load
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # Format
    img = img[:, :, ::-1]

    # When no conversion needs to be done
    if img.dtype == dtype:
        return img

    # Rescale
    img = img / np.iinfo(img.dtype).max
    img = (img * np.iinfo(dtype).max).astype(dtype)

    return img


def calc_warp_transform(
    src_img,
    dst_img,
    feature_detector=None,
    feature_matcher=None,
):

    if feature_detector is None:
        feature_detector = cv2.ORB_create()
    if feature_matcher is None:
        feature_matcher = cv2.BFMatcher()

    # Detect features
    src_kp, src_des = feature_detector.detectAndCompute(src_img, None)
    dst_kp, dst_des = feature_detector.detectAndCompute(dst_img, None)

    # Perform match
    matches = feature_matcher.match(src_des, dst_des)

    # Sort matches in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Points for the transform
    src_pts = np.array([src_kp[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2)
    dst_pts = np.array([dst_kp[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2)

    # Get the transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.)

    # Extra info dictionary
    info = {
        'src_pts': src_pts,
        'dst_pts': dst_pts,
        'mask': mask
    }

    return M, info


def validate_warp_transform(M, det_min=0.5):

    abs_det_M = np.abs(np.linalg.det(M))

    det_in_range = (
        (abs_det_M > det_min)
        and (abs_det_M < 1. / det_min)
    )

    return det_in_range


def warp_image(src_img, dst_img, M):

    # Warp the image being fit
    height, width = dst_img.shape[:2]
    warped_img = cv2.warpPerspective(src_img, M, (width, height))

    return warped_img


# def resize_image(src_img, dst_img):
# 
#     # Resize the source image
#     src_img_resized = cv2.resize(
#         src_img,
#         (dst_img.shape[1], dst_img.shape[0])
#     )


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

    required = pd.Series(required)
    required_not_in_actual = ~required.isin(actual)
    assert required_not_in_actual.sum() == 0, (
        f'Missing columns {required.loc[required_not_in_actual]}'
    )

    if isinstance(passthrough, bool):
        assert passthrough or (len(actual) == len(required)), (
            f'Expected columns {required}.\n'
            f'Got columns {list(actual)}.'
        )
    else:
        assert len(passthrough) + len(required) == len(actual), (
            f'Expecting columns {list(required) + list(passthrough)}.\n'
            f'Got columns {list(actual)}.'
        )

    return actual
