import glob
import os

import cv2
import numpy as np
import pandas as pd


import time
from abc import ABC, abstractmethod

from night_horizons import exceptions, utils


class Processor(utils.LoggerMixin, ABC):
    '''This could probably be framed as an sklearn estimator too, but let's
    not do that until necessary.

    TODO: dtype is another thing to refactor into DataIO.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        io_manager,
        image_operator,
        log_keys: list[str] = [],
        save_return_codes: list[str] = [],
        use_safe_process: bool = True,
    ):

        self.io_manager = io_manager
        self.image_operator = image_operator
        self.log_keys = log_keys
        self.save_return_codes = save_return_codes
        self.use_safe_process = use_safe_process

    def fit(self, batch_processor):
        '''Copy over fit values from the batch processor.

        TODO: We may be able to get rid of this function.

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

    def process_row(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
    ) -> pd.Series:
        '''Generally speaking, src refers to our new data, and dst refers to
        the existing data (including if the existing data was just updated
        with src in a previous row).

        TODO: A better name than "process_row" may be available, since we don't
              clarify what a row is.

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
        if self.use_safe_process:
            results = self.safe_process(i, row, resources, src, dst)
        else:
            results = self.process(i, row, resources, src, dst)
            # If it ran without error, we can assume it was a success
            results['return_code'] = 'success'

        row = self.store_results(i, row, resources, results)

        self.update_log(locals())

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
        results: dict,
    ) -> pd.Series:
        pass


class DatasetProcessor(Processor):

    def __init__(
        self,
        io_manager,
        image_operator,
        log_keys: list[str] = [],
        save_return_codes: list[str] = [],
        use_safe_process: bool = True,
        dtype: type = np.uint8,
    ):

        super().__init__(
            io_manager=io_manager,
            image_operator=image_operator,
            log_keys=log_keys,
            save_return_codes=save_return_codes,
            use_safe_process=use_safe_process,
        )
        self.dtype = dtype

    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:

        src_img = utils.load_image(
            row['filepath'],
            dtype=self.dtype,
        )

        return {'image': src_img}

    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:

        dst_img = self.get_image_from_dataset(
            resources['dataset'],
            row['x_off'],
            row['y_off'],
            row['x_size'],
            row['y_size'],
        )

        return {'image': dst_img}

    ########################################
    # Auxillary functions below

    def get_image_from_dataset(self, dataset, x_off, y_off, x_size, y_size):
        '''TODO: Refactor all IO into DataIO.
        '''

        assert x_off >= 0, 'x_off cannot be less than 0'
        assert x_off + x_size <= self.x_size_, \
            'x_off + x_size cannot be greater than self.x_size_'
        assert y_off >= 0, 'y_off cannot be less than 0'
        assert y_off + y_size <= self.y_size_, \
            'y_off + y_size cannot be greater than self.y_size_'

        # Note that we cast the input as int, in case we the input was numpy
        # integers instead of python integers.
        img = dataset.ReadAsArray(
            xoff=int(x_off),
            yoff=int(y_off),
            xsize=int(x_size),
            ysize=int(y_size),
        )
        img = img.transpose(1, 2, 0)

        self.update_log(locals())

        return img

    def save_image_to_dataset(self, dataset, img, x_off, y_off):

        img_to_save = img.transpose(2, 0, 1)
        dataset.WriteArray(
            img_to_save,
            xoff=int(x_off),
            yoff=int(y_off),
        )

        self.update_log(locals())


class DatasetUpdater(DatasetProcessor):

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:

        # Combine the images
        # TODO: image_operator is more-general,
        #       but image_blender is more descriptive
        results = self.image_operator.operate(
            src['image'],
            dst['image'],
        )
        self.update_log(self.image_operator.log)

        return {
            'blended_image': results['blended_image'],
            'src_image': src['image'],
            'dst_image': dst['image'],
        }

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ):

        # Store the image
        if results['return_code'] == 'success':
            self.save_image_to_dataset(
                resources['dataset'],
                results['blended_image'],
                row['x_off'],
                row['y_off'],
            )

        # Store the return code
        row['return_code'] = results['return_code']

        # Save some images for later debugging
        # TODO: Currently the format of the saved images is a little weird.
        if 'progress_images_dir' in self.io_manager.output_filepaths:
            progress_images_dir = (
                self.io_manager.output_filepaths['progress_images_dir']
            )
            if (
                (progress_images_dir is not None)
                and (results['return_code'] in self.save_return_codes)
            ):
                n_tests_existing = len(glob.glob(os.path.join(
                    progress_images_dir, '*_dst.tiff')))
                dst_fp = os.path.join(
                    progress_images_dir,
                    f'{n_tests_existing:06d}_dst.tiff'
                )
                src_fp = os.path.join(
                    progress_images_dir,
                    f'{n_tests_existing:06d}_src.tiff'
                )

                cv2.imwrite(src_fp, results['src_image'][:, :, ::-1])
                cv2.imwrite(dst_fp, results['dst_image'][:, :, ::-1])

                if 'blended_img' in results:
                    blended_fp = os.path.join(
                        progress_images_dir,
                        f'{n_tests_existing:06d}_blended.tiff'
                    )
                    cv2.imwrite(blended_fp, results['blended_img'][:, :, ::-1])

        return row


class DatasetRegistrar(DatasetUpdater):

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ):

        super().store_results(i, row, resources, results)

        # Store the image
        if results['return_code'] == 'success':
            self.save_image_as_new_dataset(
                img=results['warped_image'],
                x_off_orig=row['x_off'],
                y_off_orig=row['y_off'],
                x_size_orig=row['x_size'],
                y_size_orig=row['y_size'],
                image_bounds=results['new_bounds'],
            )

    def save_image_as_new_dataset(
        self,
        img,
    ):

        pass


class DatasetScorer(DatasetProcessor):

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:

        # Combine the images
        # TODO: image_operator is more-general,
        #       but image_blender is more descriptive
        results = self.image_operator.operate(
            src['image'],
            dst['image'],
        )
        self.update_log(self.image_operator.log)

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ):

        if results['return_code'] == 'success':
            row['score'] = results['score']
        else:
            row['score'] = np.nan

        row['return_code'] = results['return_code']

        return row
