"""Module for mosaicking images together."""

import os
import shutil
from typing import Tuple, Union

import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj

from ..data_io import GDALDatasetIO
from ..exceptions import OutOfBoundsError
from ..transformers import preprocessors
from ..io_manager import IOManager

from .batch import BatchProcessor

from .. import utils, raster
from . import processors

gdal.UseExceptions()


class Mosaicker(BatchProcessor):
    """This class represents a mosaicker, which is used to assemble a mosaic from
    georeferenced images. It inherits from the BatchProcessor class.
    """

    def __init__(
        self,
        io_manager: IOManager,
        crs: pyproj.CRS,
        processor: processors.Processor,
        scorer: processors.Processor = None,
        pixel_width: float = None,
        pixel_height: float = None,
        dtype: str = "uint8",
        fill_value: Union[int, float] = None,
        n_bands: int = 4,
        outline: int = 0,
        log_keys: list[str] = ["ind", "return_code"],
        passthrough: Union[list[str], bool] = True,
    ):
        """
        Initialize the Mosaicking object.

        Parameters
        ----------
        io_manager : object
            The input/output manager object, essential for saving and loading.
        processor : object
            The processor object that does the actual per-row calculations.
        scorer : object, optional
            The scorer object. Default is None.
        crs : str or pyproj.CRS, optional
            The coordinate reference system.
        pixel_width : float, optional
            The pixel width in meters. Defaults to using the pixel width of the
            input images.
        pixel_height : float, optional
            The pixel height in meters. Defaults to using the pixel height of the
            input images.
        dtype : str, optional
            The data type of the resulting mosaic. Defaults to a 0 to 255 value range.
        fill_value : int or float, optional
            Value used to fill in the empty space in the mosaic. Defaults to 0
        n_bands : int, optional
            The number of bands in the images.
        outline : int, optional
            The width of the outline to draw around each image, good for checking
            how images are combined.
        log_keys : list of str, optional
            The list of variables that are tracked.
        passthrough : list of str or bool, optional
            Which columns to pass through the transformer without altering.
            Default is True, which passes through all columns.
        """

        super().__init__(
            io_manager=io_manager,
            processor=processor,
            passthrough=passthrough,
            log_keys=log_keys,
            scorer=scorer,
        )

        # Store settings for latter use
        self.crs = crs
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.dtype = getattr(np, dtype)
        self.fill_value = fill_value
        self.n_bands = n_bands
        self.outline = outline

        self.required_columns = ["filepath"] + preprocessors.GEOTRANSFORM_COLS

    @utils.enable_passthrough
    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        i_start: Union[str, int] = "checkpoint",
        dataset: gdal.Dataset = None,
    ):
        """This method fits the Mosaicker object to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : None, optional
            The target data. Default is None.
        i_start : Union[str, int], optional
            The starting iteration. Default is 'checkpoint', which looks for any
            checkpoints and starts fresh if it doesn't find any.
        dataset : gdal.Dataset, optional
            The GDAL dataset to write to. Default is None, which creates a new dataset.

        Returns
        -------
        self : Mosaicker
            The fitted Mosaicker object.
        """
        # The fitting that's done for all image processing pipelines
        super().fit(X, y, i_start=i_start)

        # If the dataset was not passed in, load it if possible
        if (
            (self.io_manager.file_exists == "load")
            and os.path.isfile(self.io_manager.output_filepaths["mosaic"])
        ) or (self.i_start_ != 0):
            if dataset is not None:
                raise ValueError("Cannot both pass in a dataset and load a file")
            dataset = self.io_manager.open_dataset()

        # The transformer for changing between physical and pixel coordinates
        self.transformer = raster.RasterCoordinateTransformer()

        # If a dataset already exists, fit the transformer to it
        if dataset is not None:
            self.transformer.fit_to_dataset(dataset=dataset)

        # Otherwise, make a new dataset
        else:
            if self.i_start_ != 0:
                raise ValueError(
                    "Creating a new dataset, "
                    "but the starting iteration is not 0. "
                    "If creating a new dataset, should start with i = 0."
                )
            self.transformer.fit(
                X=X,
                pixel_width=self.pixel_width,
                pixel_height=self.pixel_height,
            )
            GDALDatasetIO.create(
                filepath=self.io_manager.output_filepaths["mosaic"],
                x_min=self.transformer.x_min_,
                y_max=self.transformer.y_max_,
                pixel_width=self.transformer.pixel_width_,
                pixel_height=self.transformer.pixel_height_,
                crs=self.crs,
                x_size=self.transformer.x_size_,
                y_size=self.transformer.y_size_,
                n_bands=self.n_bands,
                driver="GTiff",
            )

        # Fit the processor and scorer too
        # While this is generic and expected for the majority of
        # BatchProcessors, it cannot be part of self.fit because it has to be
        # called after the rest of the fitting is done.
        self.processor.fit(self)
        if self.scorer is not None:
            self.scorer.fit(self)

        return self

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Preprocessing required before doing the full loop.
        For mosaickers this includes putting the
        coordinates in the pixel-based frame of the mosaic.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing the preprocessed input data
            and a dictionary of resources.
        """

        X_t = self.transformer.transform_to_pixel(X)

        # Get the dataset
        resources = {
            "dataset": self.io_manager.open_dataset(),
            "coord_transformer": self.transformer,
        }

        if pyproj.CRS(resources["dataset"].GetProjection()) != self.crs:
            raise ValueError("Mosaic dataset has the wrong CRS.")

        return X_t, resources

    def postprocess(self, X_t: pd.DataFrame, resources: dict) -> pd.DataFrame:
        """Postprocessing after the full loop.
        This focuses on cleaning up any resources used during the processing,
        and finalizing the output data.

        Parameters
        ----------
        X_t : pd.DataFrame
            The processed input data.
        resources : dict
            A dictionary of resources used during the processing.

        Returns
        -------
        pd.DataFrame
            The postprocessed input data.
        """

        # Close out the dataset
        resources["dataset"].FlushCache()
        resources["dataset"] = None

        return X_t


class SequentialMosaicker(Mosaicker):
    """Class for mosaicking images together in a sequential manner.
    Doing so allows us to build up a mosaic from a series of images, one at a time,
    and reference the images in the process.

    Unfortunately, doing the referencing this way is inaccurate, because the error
    will grow with each image added.
    """

    def __init__(
        self,
        io_manager: IOManager,
        crs: pyproj.CRS,
        processor: processors.Processor,
        mosaicker_train: Mosaicker,
        scorer: processors.Processor = None,
        progress_images_subdir: str = "progress_images",
        save_return_codes: list[str] = [],
        memory_snapshot_freq: int = 10,
        pixel_width: float = None,
        pixel_height: float = None,
        fill_value: Union[int, float] = None,
        dtype: str = "uint8",
        n_bands: int = 4,
        passthrough: Union[bool, list[str]] = True,
        outline: int = 0,
        log_keys: list[str] = ["i", "ind", "return_code", "abs_det_M"],
    ):
        """
        Initialize the SequentialMosaicker.

        Parameters
        ----------
        io_manager : IOManager
            The IOManager object used for input/output operations.
        crs : pyproj.CRS
            The coordinate reference system (CRS) of the mosaic.
        processor : processors.Processor
            The processor object used for image processing.
        mosaicker_train : Mosaicker
            The Mosaicker object used for the starting mosaic of referenced images.
        scorer : processors.Processor, optional
            The scorer object used for scoring the mosaic, by default None.
        progress_images_subdir : str, optional
            The subdirectory name for storing images of the mosaic under certain return
            codes.
        save_return_codes : list[str], optional
            The list of return codes to save to the progress images dir,
            by default []. These are the return codes from the processor object,
            and indicate success, failure, etc.
        memory_snapshot_freq : int, optional
            How often to take a snapshot of the memory usage, by default 10.
            Memory snapshots are only taken if the key "snapshot" is in log_keys.
        pixel_width : float, optional
            The pixel width in meters. Defaults to using the pixel width of the
            input images.
        pixel_height : float, optional
            The pixel height in meters. Defaults to using the pixel height of the
            input images.
        fill_value : int or float, optional
            Value used to fill in the empty space in the mosaic. Defaults to 0
        dtype : str, optional
            The data type of the resulting mosaic. Defaults to a 0 to 255 value range.
        n_bands : int, optional
            The number of bands in the images.
        outline : int, optional
            The width of the outline to draw around each image, good for checking
            how images are combined.
        log_keys : list[str], optional
            The list of keys to log, by default
            ["i", "ind", "return_code", "abs_det_M"].

        Returns
        -------
        None
        """

        super().__init__(
            io_manager=io_manager,
            processor=processor,
            scorer=scorer,
            crs=crs,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            fill_value=fill_value,
            dtype=dtype,
            n_bands=n_bands,
            passthrough=passthrough,
            outline=outline,
            log_keys=log_keys,
        )

        self.mosaicker_train = mosaicker_train
        self.progress_images_subdir = progress_images_subdir
        self.save_return_codes = save_return_codes
        self.memory_snapshot_freq = memory_snapshot_freq

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        X_train: pd.DataFrame = None,
        dataset: gdal.Dataset = None,
        i_start: Union[int, str] = "checkpoint",
    ):
        """Create the initial mosaic out of the referenced images.
        The size of the mosaic is determined by the estimated spatial extent of the
        referenced + unreferenced images.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the unreferenced images.
        y : None, optional
            The target variable (not used in this method), but required
            for compatibility.
        X_train : pd.DataFrame, optional
            The input DataFrame containing the referenced images
            to build a base mosaic.
        dataset : gdal.Dataset, optional
            The GDAL dataset to use for creating the mosaic. A new one is created by
            default.
        i_start : int or str, optional
            The starting index for creating the mosaic.
            If "checkpoint", it starts from a checkpoint file if available.

        Returns
        -------
        self : Mosaicker
            Returns the Mosaicker object itself.

        Raises
        ------
        OutOfBoundsError
            If some of the fitted referenced images are out of bounds.
        """

        assert (
            X_train is not None
        ), "Must pass X_train (referenced images to build a base mosaic)"

        # Create the initial mosaic.
        # This is fit to the both the training and the search regions for the
        # actual data.
        X_for_fit = pd.concat([X, X_train])
        super().fit(X=X_for_fit, dataset=dataset, i_start=i_start)

        # Create the initial mosaic, if not starting from a checkpoint file
        if self.i_start_ == 0:
            dataset = self.io_manager.open_dataset()
            try:
                self.mosaicker_train.fit_transform(
                    X=X_train, dataset=dataset, i_start=0
                )
            except OutOfBoundsError as e:
                raise OutOfBoundsError(
                    "Some of the fitted referenced images are out of bounds. "
                    "Consider increasing the 'padding' in approx_y."
                ) from e

            # Close, to be safe
            dataset.FlushCache()
            dataset = None

            # Save the fit mosaic, pre-prediction
            mosaic_fp = self.io_manager.output_filepaths["mosaic"]
            shutil.copy(mosaic_fp, mosaic_fp.replace(".tiff", "_fit.tiff"))

        return self

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Preprocessing required before doing the full loop.
        This includes setting up the search zone and transforming to pixel coordinates.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for preprocessing.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing the preprocessed data and a dictionary of resources.
        """

        # Get state of output data
        if self.checkpoint_state_ is None:
            X_t = self.get_search_zone(X)
            X_t = self.transformer.transform_to_pixel(X_t)

            # And the logs too
            self.logs = []
        else:
            X_t = self.checkpoint_state_["y_pred"]

        # Get the dataset
        resources = {
            "dataset": self.io_manager.open_dataset(),
            "transformer": self.transformer,
        }

        return X_t, resources

    def postprocess(self, y_pred: pd.DataFrame, resources: dict) -> pd.DataFrame:
        """Postprocess the data. This includes converting back to physical
        coordinates, saving logs, and ensuring nothing is left in memory.

        Parameters
        ----------
        y_pred : pd.DataFrame
            The predicted values from the processor.

        resources : dict
            A dictionary of additional resources.

        Returns
        -------
        pd.DataFrame
            The postprocessed DataFrame.
        """

        # Convert to pixels
        y_pred = self.transformer.transform_to_physical(y_pred)
        y_pred["pixel_width"] = self.transformer.pixel_width_
        y_pred["pixel_height"] = self.transformer.pixel_height_
        y_pred["x_center"] = 0.5 * (y_pred["x_min"] + y_pred["x_max"])
        y_pred["y_center"] = 0.5 * (y_pred["y_min"] + y_pred["y_max"])

        y_pred.to_csv(self.io_manager.output_filepaths["y_pred"])

        # Store log
        log_df = pd.DataFrame(self.logs)
        log_df.to_csv(self.io_manager.output_filepaths["log"])

        # Flush data to disk
        resources["dataset"].FlushCache()
        resources["dataset"] = None

        return y_pred

    def get_search_zone(self, X: pd.DataFrame) -> pd.DataFrame:
        """Modify the estimated extent of the images to allow for a search zone.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the estimated extent of the images.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with the estimated extent of the images
            expanded to include a search zone.
        """

        X["x_min"] = X["x_min"] - X["padding"]
        X["x_max"] = X["x_max"] + X["padding"]
        X["y_min"] = X["y_min"] - X["padding"]
        X["y_max"] = X["y_max"] + X["padding"]

        return X
