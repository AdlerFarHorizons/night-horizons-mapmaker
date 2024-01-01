from abc import ABC, abstractmethod
import time
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import utils, exceptions


class BaseBatchProcesser(
    utils.LoopLoggerMixin,
    TransformerMixin,
    BaseEstimator,
    ABC
):

    def __init__(self, row_processor, passthrough, log_keys):
        '''
        Parameters
        ----------
        Returns
        -------
        '''

        self.row_processor = row_processor
        self.passthrough = passthrough
        self.log_keys = log_keys

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
        self.out_dir_, self.filepath_ = self.io_manager.prepare_filetree()

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

        # This checks both the input and the state of the class
        # (e.g. is it fitted?)
        # This also makes X a copy,
        # so we don't accidentally modify the original
        X = self.validate_readiness(X)

        # TODO: We could avoid passing around the log filepath here, and
        #       keep it as an attribute instead...
        #       One nice thing about this as is is that we don't have to
        #       go digging for where the log is saved.
        if 'log' in self.io_manager.aux_filepaths_:
            self.log_filepath_ = self.io_manager.aux_filepaths_['log']
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
        X_t, resources = self.preprocess(X)

        # Main loop
        for i, ind in enumerate(tqdm.tqdm(X_t.index, ncols=80)):

            # Go to the right loop
            if i < self.i_start_:
                continue

            row = X.loc[ind]
            row = self.row_processor.transform_row(i, row, resources)
            X_t.loc[ind] = row

            # Checkpoint
            resources['dataset'] = self.io_manager.save_to_checkpoint(
                i,
                resources['dataset'],
            )

            # Update and save the log
            # TODO: We probably don't have to write every loop...
            self.logs.append(self.row_processor.log)
            if hasattr(self, 'log_filepath_'):
                self.write_log(self.log_filepath_)

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

        return X

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        pass

    @abstractmethod
    def postprocess(self, X: pd.DataFrame, resources: dict) -> pd.DataFrame:
        pass


class BaseRowProcessor(utils.LoggerMixin, ABC):
    '''This could probably be framed as an sklearn estimator too, but let's
    not do that until necessary.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(self, image_processor, log_keys: list[str] = []):

        self.image_processor = image_processor
        self.log_keys = log_keys

    def fit(self, batch_processor):
        '''Copy over fit values from the batch processor.

        Parameters
        ----------
        Returns
        -------
        '''

        for attr_name in batch_processor.__dir__():
            # Fit variables have names ending with an underscore
            if (attr_name[-1] != '_') or (attr_name[-2:] == '__'):
                continue
            attr_value = getattr(batch_processor, attr_name)
            setattr(self, attr_name, attr_value)

        return self

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

        # Get data
        src = self.get_src(i, row, resources)
        dst = self.get_dst(i, row, resources)

        # Main function that changes depending on parent class
        results = self.safe_process(i, row, resources, src, dst)

        self.store_results(i, row, resources, results)

        return row

    def safe_process(self, i, row, resources, src, dst):
        '''
        Parameters
        ----------
        Returns
        -------
            results:
                blended_img: Combined image. Not always returned.
                M: Homography transform. Not always returned.
                src_kp: Keypoints for the src image. Not always returned.
                src_des: KP descriptors for the src image. Not always returned.
                duration: Time spent.
        '''

        start = time.time()

        results = {}
        return_code = 'not_set'
        try:
            results = self.process(i, row, resources, src, dst)
            return_code = 'success'
        except cv2.error:
            return_code = 'opencv_err'
        except exceptions.HomographyTransformError:
            return_code = 'bad_det'
        except exceptions.SrcDarkFrameError:
            return_code = 'dark_frame'
        except exceptions.DstDarkFrameError:
            return_code = 'dst_dark_frame'
        except np.linalg.LinAlgError:
            return_code = 'linalg_err'
        finally:
            duration = time.time() - start
            results['duration'] = duration
            results['return_code'] = return_code

            return results

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
    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        result: dict,
    ):
        pass
