import tracemalloc
from typing import Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import utils
from .processors import Processor


class BatchProcessor(
    utils.LoopLoggerMixin,
    TransformerMixin,
    BaseEstimator,
):

    def __init__(
        self,
        processor: Processor,
        passthrough,
        log_keys,
        scorer: Processor = None,
    ):
        '''
        Parameters
        ----------
        Returns
        -------
        '''

        self.processor = processor
        self.passthrough = passthrough
        self.log_keys = log_keys
        self.scorer = scorer

    @utils.enable_passthrough
    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        i_start: Union[str, int] = 'checkpoint',
    ):
        '''The main thing the fitting does is create an empty dataset to hold
        the mosaic.

        Parameters
        ----------
        X
            A dataframe containing the bounds of each added image.

        y
            Empty.

        Returns
        -------
        self
            Returns self
        '''

        # Save the settings used for fitting
        # Must be done after preparing the filetree to have a save location
        self.io_manager.save_settings(self)

        # Start from checkpoint, if available
        if i_start == 'checkpoint':
            self.i_start_, self.checkpoint_state_ = \
                self.io_manager.search_and_load_checkpoint()
        else:
            self.i_start_ = i_start
            self.checkpoint_state_ = None

        return self

    @utils.enable_passthrough
    def transform(
        self,
        X: pd.DataFrame,
        y=None,
    ):

        return self.batch_process(
            self.processor,
            X,
            y,
        )

    def predict(
        self,
        X: pd.DataFrame,
    ):
        '''Transform and predict perform the same process here.
        Transform is appropriate for image processing as an intermediate step.
        Predict is appropriate for image processing as the final step.
        '''

        return self.transform(X)

    def score(
        self,
        X: pd.DataFrame,
        y=None,
    ):

        return self.batch_process(
            self.scorer,
            X,
            y,
        )

    def batch_process(
        self,
        processor: Processor,
        X: pd.DataFrame,
        y=None,
    ):

        # This checks both the input and the state of the class
        # (e.g. is it fitted?)
        # This also makes X a copy,
        # so we don't accidentally modify the original
        X = self.validate_readiness(X)

        # TODO: We could avoid passing around the log filepath here, and
        #       keep it as an attribute instead...
        #       One nice thing about this as is is that we don't have to
        #       go digging for where the log is saved.
        if 'log' in self.io_manager.output_filepaths:
            self.log_filepath_ = self.io_manager.output_filepaths['log']
            self.start_logging(
                i_start=self.i_start_,
                log_filepath=self.log_filepath_,
            )
        else:
            self.start_logging()

        # Resources contains global variables that will be available
        # throughout image processing.
        # TODO: I may be able to get away without resources. Resources
        #       originally existed because I thought saving dataset as
        #       an attribute (for mosaicking) was creating a memory leak.
        #       However, when I debugged that much of the issue actually came
        #       from saving massive objects (all the features) to the log
        #       and duplicating them.
        Z_out, resources = self.preprocess(X)

        # Start memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.start()
            start = tracemalloc.take_snapshot()
            self.log['starting_snapshot'] = start

        # Main loop
        for i, ind in enumerate(tqdm.tqdm(X.index, ncols=80)):

            # Go to the right loop
            if i < self.i_start_:
                continue

            # We make a copy of row so that we don't modify the original
            row = X.loc[ind].copy()

            # Process the row
            row = processor.process_row(i, row, resources)
            Z_out.loc[ind] = row

            # Snapshot the memory usage
            log = processor.log
            if 'snapshot' in self.log_keys:
                if i % self.memory_snapshot_freq == 0:
                    log['snapshot'] = tracemalloc.take_snapshot()

            # Checkpoint
            resources['dataset'] = self.io_manager.save_to_checkpoint(
                i,
                resources['dataset'],
            )

            # Update and save the log
            # TODO: We probably don't have to write every loop...
            self.logs.append(log)
            if hasattr(self, 'log_filepath_'):
                self.write_log(self.log_filepath_)

        # Stop memory tracing
        if 'snapshot' in self.log_keys:
            tracemalloc.stop()

        Z_out = self.postprocess(Z_out, resources)

        return Z_out


    def validate_readiness(self, X: pd.DataFrame):
        '''Pre-transform validation.

        Parameters
        ----------
        Returns
        -------
        '''

        # This is where X is copied too
        X = utils.check_df_input(
            X,
            self.required_columns,
        )

        # Check if fit had been called
        check_is_fitted(self, 'i_start_')

        return X

    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        return X, {}

    def postprocess(self, X: pd.DataFrame, resources: dict) -> pd.DataFrame:
        return X


