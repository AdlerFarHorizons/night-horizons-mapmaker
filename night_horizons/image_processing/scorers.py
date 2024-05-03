"""Module for scoring image processing results.
"""

import cv2
import numpy as np
import pandas as pd
import pyproj

from .operators import BaseImageOperator
from .processors import Processor, DatasetProcessor
from ..data_io import GDALDatasetIO


class SimilarityScoreOperator(BaseImageOperator):
    """Class for scoring based on the similarity of two images."""

    def __init__(
        self,
        allow_resize: bool = True,
        compare_nonzero: bool = True,
        tm_metric=cv2.TM_CCOEFF_NORMED,
        log_keys: list[str] = [],
        acceptance_threshold: float = 0.99,
    ) -> None:
        """Initialize the SimilarityScoreOperator object.

        Parameters
        ----------
        allow_resize : bool, optional
            Flag indicating whether resizing of images is allowed during comparison.
            Defaults to True.
        compare_nonzero : bool, optional
            Flag indicating whether only non-zero pixels should be compared during
            image comparison. Defaults to True.
        tm_metric : int, optional
            The template matching metric to be used. Defaults to cv2.TM_CCOEFF_NORMED.
        log_keys : list of str, optional
            List of local variables to be logged during image comparison.
            Defaults to an empty list.
        acceptance_threshold : float, optional
            The threshold value for accepting a match. Defaults to 0.99.
        """
        self.allow_resize = allow_resize
        self.compare_nonzero = compare_nonzero
        self.tm_metric = tm_metric
        self.log_keys = log_keys
        self.acceptance_threshold = acceptance_threshold

    def operate(self, src_img: np.ndarray, dst_img: np.ndarray) -> dict:
        """Compare two images using template matching.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to compare.

        dst_img : np.ndarray
            The destination image to compare.

        Returns
        -------
        dict
            A dictionary containing the score of the comparison.

        Raises
        ------
        ValueError
            If the destination image has fewer channels than the source image or if
            the images have different shapes and resizing is not allowed.
        """

        if src_img.shape[-1] != dst_img.shape[-1]:
            if dst_img.shape[-1] < src_img.shape[-1]:
                raise ValueError(
                    "Destination image must have as many or more channels "
                    "than source image."
                )
            dst_img = dst_img[..., : src_img.shape[-1]]

        if src_img.shape != dst_img.shape:
            if not self.allow_resize:
                raise ValueError("Images must have the same shape.")
            src_img = cv2.resize(src_img, (dst_img.shape[1], dst_img.shape[0]))

        if self.compare_nonzero:
            empty_src = np.isclose(src_img.sum(axis=2), 0.0)
            empty_dst = np.isclose(dst_img.sum(axis=2), 0.0)
            either_empty = np.logical_or(empty_src, empty_dst)
            src_img[either_empty] = 0
            dst_img[either_empty] = 0

        r = cv2.matchTemplate(src_img, dst_img, self.tm_metric)[0][0]

        return {"score": r}

    def assert_equal(self, src_img: np.ndarray, dst_img: np.ndarray):
        """Assert that two images are equal based on their similarity score.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to compare.
        dst_img : np.ndarray
            The destination image to compare.
        """
        results = self.operate(src_img, dst_img)
        assert (
            results["score"] > self.acceptance_threshold
        ), f'Images have a score of {results["score"]}'


class DatasetScorer(DatasetProcessor):
    """Class for scoring image processing results in the context of a gdal dataset."""

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        """
        Process the images and return the results.

        Parameters
        ----------
        i : int
            The index of the image being processed.
        row : pd.Series
            The row containing the image metadata.
        resources : dict
            Additional resources for image processing.
        src : dict
            The source image dictionary.
        dst : dict
            The destination image dictionary.

        Returns
        -------
        dict
            The results of the image processing.
        """

        # Combine the images
        results = self.image_operator.operate(
            src["image"],
            dst["image"],
        )
        self.update_log(self.image_operator.log)

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ) -> pd.Series:
        """Save the results of image processing to the metadata.

        Parameters
        ----------
        i : int
            The index of the image being processed.
        row : pd.Series
            The metadata row corresponding to the image being processed.
        resources : dict
            Additional resources used during image processing.
        results : dict
            The results of the image processing.

        Returns
        -------
        pd.Series
            The updated metadata row with the stored results.
        """

        if results["return_code"] == "success":
            row["score"] = results["score"]
        else:
            row["score"] = np.nan

        row["return_code"] = results["return_code"]

        return row


class ReferencedImageScorer(Processor):
    """Class for scoring similarity between two referenced images."""

    def __init__(self, crs: pyproj.CRS = None, *args, **kwargs):
        """
        Initialize the ReferencedImageScorer object.

        Parameters
        ----------
        crs : pyproj.CRS, optional
            The coordinate reference system (CRS) to be used, by default None.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        self.crs = crs

        super().__init__(*args, **kwargs)

    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:
        """
        Get the source referenced image for scoring.

        Parameters
        ----------
        i : int
            The index of the row.
        row : pd.Series
            The row containing the filepath.
        resources : dict
            Additional resources.

        Returns
        -------
        dict
            A dictionary containing the source dataset.
        """

        src_dataset = GDALDatasetIO.load(
            row["filepath"],
            crs=self.crs,
        )

        return {"dataset": src_dataset}

    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:
        """
        Get the destination referenced image for scoring.

        Parameters
        ----------
        i : int
            The index of the row.
        row : pd.Series
            The row containing the filepath.
        resources : dict
            Additional resources.

        Returns
        -------
        dict
            A dictionary containing the destination dataset.
        """

        if pd.isna(row["output_filepath"]):
            return {"dataset": None}

        dst_dataset = GDALDatasetIO.load(
            row["output_filepath"],
            crs=self.crs,
        )

        return {"dataset": dst_dataset}

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        """
        Calculate the scores.

        Parameters
        ----------
        i : int
            The index of the image.
        row : pd.Series
            The row containing the image data.
        resources : dict
            Additional resources for processing.
        src : dict
            The source dataset containing the image data.
        dst : dict
            The destination dataset for the processed image data.

        Returns
        -------
        dict
            A dictionary containing the calculated scores and differences.
        """

        if dst["dataset"] is None:
            return {}

        src_x_bounds, src_y_bounds, src_pixel_width, src_pixel_height = (
            GDALDatasetIO.get_bounds_from_dataset(src["dataset"])
        )
        dst_x_bounds, dst_y_bounds, dst_pixel_width, dst_pixel_height = (
            GDALDatasetIO.get_bounds_from_dataset(dst["dataset"])
        )

        results = {
            "x_size_diff": (dst["dataset"].RasterXSize - src["dataset"].RasterXSize),
            "y_size_diff": (dst["dataset"].RasterYSize - src["dataset"].RasterYSize),
            "x_min_diff": dst_x_bounds[0] - src_x_bounds[0],
            "x_max_diff": dst_x_bounds[1] - src_x_bounds[1],
            "y_min_diff": dst_y_bounds[0] - src_y_bounds[0],
            "y_max_diff": dst_y_bounds[1] - src_y_bounds[1],
            "pixel_width_diff": dst_pixel_width - src_pixel_width,
            "pixel_height_diff": dst_pixel_height - src_pixel_height,
        }
        results["center_diff"] = 0.5 * np.sqrt(
            (results["x_min_diff"] + results["x_max_diff"]) ** 2.0
            + (results["y_min_diff"] + results["y_max_diff"]) ** 2.0
        )

        # If we actually want to compare the image values
        if self.image_operator is not None:
            results["image_operator_score"] = self.image_operator.operate(
                src["dataset"].ReadAsArray(),
                dst["dataset"].ReadAsArray(),
            )

        results["score"] = results["center_diff"]

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ) -> pd.Series:
        """
        Store the results of image processing for a given row.

        Parameters
        ----------
        i : int
            The index of the row being processed.
        row : pd.Series
            The row containing the image processing results.
        resources : dict
            Additional resources used during image processing.
        results : dict
            The dictionary containing the image processing results.

        Returns
        -------
        pd.Series
            The updated row with the image processing results stored.
        """

        # Combine
        for key, item in results.items():
            row[key] = item

        return row
