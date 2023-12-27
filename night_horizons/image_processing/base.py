from abc import abstractmethod
from typing import Tuple, Union

import cv2
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import utils


class BaseProcessor(utils.LoopLoggerMixin, TransformerMixin, BaseEstimator):

    def __init__(self, file_manager, logger, row_processor, log_keys=[]):

        self.file_manager = file_manager
        self.logger = logger
        self.row_processor = row_processor

        super().__init__(log_keys=log_keys)

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

        # Make output directories, get filepaths, load dataset (if applicable)
        self.out_dir_, self.filepath_ = self.file_manager.prepare_filetree()

        # Save the settings used for fitting
        # Must be done after preparing the filetree to have a save location
        self.file_manager.save_settings(self)

        # Start from checkpoint, if available
        if i_start == 'checkpoint':
            self.i_start_, self.checkpoint_state_ = \
                self.file_manager.search_and_load_checkpoint()
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

        # This checks both the input and the state of the class
        # (e.g. is it fitted?)
        # This also makes X a copy,
        # so we don't accidentally modify the original
        X = self.validate_readiness(X)

        # TODO: We could avoid passing around the log filepath here, and
        #       keep it as an attribute instead...
        #       One nice thing about this is that we don't have to go digging
        #       for where the log is saved.
        log_filepath = self.file_manager.aux_filepaths['log']
        self.start_logging(
            i_start=self.i_start_,
            log_filepath=log_filepath,
        )

        # Resources contains global variables that will be available
        # throughout image processing.
        X_t, resources = self.preprocess(X)

        # If verbose, add a progress bar.
        if self.verbose:
            iterable = tqdm.tqdm(X_t.index, ncols=80)
        else:
            iterable = X_t.index

        # Main loop
        for i in range(len(iterable)):

            # Go to the right loop
            if i < self.i_start_:
                continue

            row = X.iloc[i]
            row = self.row_processer.transform_row(i, row, resources)
            X_t.iloc[i] = row

            # Checkpoint
            resources['dataset'] = self.file_manager.save_to_checkpoint(
                i,
                resources['dataset'],
            )

            # Update and save the log
            # TODO: We probably don't have to write every loop...
            self.logs.append(self.row_processor.log)
            self.write_log(log_filepath)

        X_t = self.postprocess(X_t, resources)

        return X_t

    def predict(
        self,
        X: pd.DataFrame,
    ):
        '''Transform and predict perform the same process here.
        Transform is appropriate for image processing as an intermediate step.
        Predict is appropriate for image processing as the final step.
        '''

        return self.transform(X)

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
        check_is_fitted(self, 'out_dir_')

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        pass

    @abstractmethod
    def postprocess(self, X: pd.DataFrame, resources: dict) -> pd.DataFrame:
        pass


class BaseRowProcessor(utils.LoggerMixin):
    '''This could probably be framed as an sklearn estimator too, but let's
    not do that until necessary.

    Parameters
    ----------
    Returns
    -------
    '''

    def transform_row(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
    ) -> pd.Series:
        '''Generally speaking, src refers to our new data, and dst refers to
        the existing data (including if the existing data was just updated
        with src in a previous row).

        Parameters
        ----------
        Returns
        -------
        '''

        self.start_logging()

        # Get data
        src = self.get_src(i, row, resources)
        dst = self.get_dst(i, row, resources)

        # Main function that changes depending on parent class
        result = self.process(i, row, resources, src, dst)

        self.store_result(i, row, resources, result)

        return row

    @abstractmethod
    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:
        pass

    @abstractmethod
    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:
        pass

    @abstractmethod
    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        pass

    @abstractmethod
    def store_result(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        result: dict,
    ):
        pass
