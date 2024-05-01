"""Module for handling batch processing of images.
"""

import tracemalloc
from typing import Tuple, Union

import numpy as np
import os
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import utils
from .processors import Processor
from ..io_manager import IOManager

# Set up the logger
LOGGER = utils.get_logger(__name__)


class BatchProcessor(
    utils.LoopLoggerMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Base class for processing images in batches."""

    def __init__(
        self,
        io_manager: IOManager,
        processor: Processor,
        log_keys: list[str] = ["ind", "return_code"],
        passthrough: Union[list[str], bool] = True,
        scorer: Processor = None,
    ):
        """
        Initialize the BatchProcessor object.

        Parameters
        ----------
        processor : Processor
            The processor object used for batch processing.

        log_keys : list[str], optional
            The list of keys to include in the log.

        passthrough : Union[list[str], bool], optional
            Determines whether to pass the input data through the processor.
            If True, the input data will be passed through the processor.
            If False, the input data will not be passed through the processor.
            If a list of strings is provided, only the specified keys will be
            passed through the processor.

        scorer : Processor, optional
            The scorer object used for scoring the processed data.

        Passthrough is currently True by default. This is less likely to
        cause unexpected errors, but slightly more likely to cause uncaught
        unexpected behavior.
        """
        self.io_manager = io_manager
        self.processor = processor
        self.passthrough = passthrough
        self.log_keys = log_keys
        self.scorer = scorer

    @utils.enable_passthrough
    def fit(
        self,
        X: pd.DataFrame = None,
        y=None,
        i_start: Union[str, int] = "checkpoint",
    ) -> "BatchProcessor":
        """
        Fit the BatchProcessor model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The input data. Default is None.
        y : Any, optional
            The target data. Default is None.
        i_start : Union[str, int], optional
            The starting index for fitting. Default is 'checkpoint'.

        Returns
        -------
        BatchProcessor
            The fitted BatchProcessor model.
        """
        # Save the settings used for fitting
        self.io_manager.save_settings(self)

        # Start from checkpoint, if available
        if i_start == "checkpoint":
            self.i_start_, self.checkpoint_state_ = (
                self.io_manager.search_and_load_checkpoint()
            )
        else:
            self.i_start_ = i_start
            self.checkpoint_state_ = None

        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """Transform the input data using the BatchProcessor.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to be transformed.

        y : Any, optional
            The target data. Default is None.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        # Memory information
        mem = psutil.virtual_memory()
        LOGGER.info(
            "At start of transform: "
            f"{mem.available / 1024**3.:.2g} of "
            f"{mem.total / 1024**3.:.2g} GB available "
            f"({mem.percent}% used)"
        )

        # This checks both the input and the state of the class
        # (e.g. is it fitted?)
        # This also makes X a copy,
        # so we don't accidentally modify the original
        X = self.validate_readiness(X)

        return self.batch_process(
            self.processor,
            X,
            y,
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform and predict are the same functions.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to be transformed and predicted.

        Returns
        -------
        pd.DataFrame
            The transformed and predicted data.
        """
        return self.transform(X)

    def score(
        self,
        X: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """Calculate the scores for the given input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to be scored.
        y : Optional
            The true-value output.

        Returns
        -------
        pd.DataFrame
            The scored data.
        """
        # Make a copy so we don't directly alter the data
        X = X.copy()

        X_out = self.batch_process(
            self.scorer,
            X,
            y,
        )

        return X_out

    def batch_process(
        self,
        processor: Processor,
        X: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """Batch process the data using a given processor.

        Parameters
        ----------
        processor : Processor
            The processor object used to process each row of the data.
        X : pd.DataFrame
            The input data to be processed.
        y : None, optional
            The target data (default is None).

        Returns
        -------
        pd.DataFrame
            The processed output data.
        """

        LOGGER.info("Starting batch processing...")

        if "log" in self.io_manager.output_filepaths:
            self.log_filepath_ = self.io_manager.output_filepaths["log"]
            self.start_logging(
                i_start=self.i_start_,
                log_filepath=self.log_filepath_,
            )
        else:
            self.start_logging()

        LOGGER.info("Starting batch processing:preprocessing...")

        # Resources contains global variables that will be available
        # throughout image processing.
        X_t, resources = self.preprocess(X)

        # Start memory tracing
        if "snapshot" in self.log_keys:
            tracemalloc.start()
            start = tracemalloc.take_snapshot()
            self.log["starting_snapshot"] = start

        LOGGER.info("Starting batch processing:main loop...")

        # Main loop
        Z_out = X_t.copy()
        for i, ind in enumerate(tqdm.tqdm(X.index, ncols=80)):

            # Go to the right loop
            if i < self.i_start_:
                continue

            # We make a copy of row so that we don't modify the original
            row = X_t.loc[ind].copy()

            LOGGER.info(f"Processing row {i}...")

            # Process the row
            row = processor.process_row(i, row, resources)

            LOGGER.info(f"Postprocessing row {i}...")

            # Combine
            Z_out = row.to_frame().T.combine_first(Z_out)

            # Snapshot the memory usage
            log = processor.log
            if "snapshot" in self.log_keys:
                if i % self.memory_snapshot_freq == 0:
                    log["snapshot"] = tracemalloc.take_snapshot()

            LOGGER.info(f"Checkpointing row {i} in batch processing:main loop...")

            # Checkpoint
            resources["dataset"] = self.io_manager.save_to_checkpoint(
                i,
                resources["dataset"],
                y_pred=Z_out,
            )
            # Clean old checkpoint
            self.io_manager.prune_checkpoints()

            # Update and save the log
            log = self.update_log(locals(), log)
            self.logs.append(log)
            if hasattr(self, "log_filepath_"):
                self.write_log(self.log_filepath_)

        # It's possible for the data to get scrambled during processing,
        # so we sort it before returning it.
        Z_out = Z_out.loc[X.index]

        # Stop memory tracing
        if "snapshot" in self.log_keys:
            tracemalloc.stop()

        LOGGER.info("Starting batch processing:postprocessing...")

        Z_out = self.postprocess(Z_out, resources)

        return Z_out

    def validate_readiness(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pre-transform validation.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be validated.

        Returns
        -------
        pd.DataFrame
            The validated input DataFrame.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """

        # This is where X is copied too
        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Check if fit had been called
        check_is_fitted(self, "i_start_")

        return X

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Preprocesses the input data. This method is intended to be overwritten
        by subclasses.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to be preprocessed.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing the preprocessed data and an empty dictionary.
        """

        return X, {}

    def postprocess(self, X: pd.DataFrame, resources: dict) -> pd.DataFrame:
        """
        Postprocesses the input DataFrame. This method is intended to be
        overwritten by subclasses.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be postprocessed.

        resources : dict
            A dictionary of additional resources.

        Returns
        -------
        pd.DataFrame
            The postprocessed DataFrame.
        """

        return X
