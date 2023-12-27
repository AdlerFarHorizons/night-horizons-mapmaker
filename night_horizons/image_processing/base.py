from abc import abstractmethod
from typing import Tuple

import cv2
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from .. import utils


class BaseProcessor(utils.LoggerMixin, TransformerMixin, BaseEstimator):

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

        self.initialize_logs()

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
            row = X.iloc[i]
            row = self.transform_row(i, row, resources)
            X_t.iloc[i] = row

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

    def transform_row(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
    ) -> pd.Series:

        # Get data
        src = self.get_src(i, row, resources)
        dst = self.get_dst(i, row, resources)

        # Resize the source image
        src_img_resized = cv2.resize(
            src_img,
            (dst_img.shape[1], dst_img.shape[0])
        )

        # Combine the images
        blended_img = utils.blend_images(
            src_img=src_img_resized,
            dst_img=dst_img,
            fill_value=self.fill_value,
            outline=self.outline,
        )

        # Store the image
        self.save_image(
            resources['dataset'],
            blended_img,
            row['x_off'],
            row['y_off'],
        )

        # Update the log
        log = self.update_log(locals(), target={})
        self.logs.append(log)

        # Checkpoint
        resources['dataset'] = self.file_manager.save_to_checkpoint(
            i,
            resources['dataset'],
            logs=self.logs,
        )

        return row

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

    def initialize_logs(self):
        '''Prepare the two types of logging: self.log for one-time variables,
        and self.logs for per-image variables.

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
        if self.checkpoint_data_ is None:
            self.logs = []
        else:
            self.logs = self.checkpoint_data_['logs']

    @abstractmethod
    def preprocess(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        pass



    @abstractmethod
    def postprocess(self, X: pd.DataFrame, resources: dict) -> pd.DataFrame:
        pass
