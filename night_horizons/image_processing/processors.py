"""Module for processing a single "row", aka a single image, in the context of
other images and metadata.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import pyproj


import time
from abc import ABC, abstractmethod
from osgeo import gdal

from night_horizons import exceptions, utils
from night_horizons.data_io import ImageIO, ReferencedImageIO
from night_horizons.io_manager import IOManager
from night_horizons.transformers.raster import RasterCoordinateTransformer
from .operators import BaseImageOperator


# Set up the logger
LOGGER = utils.get_logger(__name__)


class Processor(utils.LoggerMixin, ABC):
    '''Processor is an abstract class that defines the structure of a processor.

    This could probably be framed as an sklearn estimator too, but let's
    not do that until necessary.
    '''

    def __init__(
        self,
        io_manager: IOManager,
        image_operator: BaseImageOperator,
        log_keys: list[str] = [],
        save_return_codes: list[str] = [],
        use_safe_process: bool = True,
    ):
        """
        Initialize a Processor object.

        Parameters
        ----------
        io_manager : IOManager
            The IOManager object responsible for managing input/output operations.
        image_operator : BaseImageOperator
            The image operator object used for the operations at the core of
            processing.
        log_keys : list[str], optional
            The list of local variables to log in tabular form,
            by default an empty list.
        save_return_codes : list[str], optional
            The list of return codes to save to the progress images directory,
            by default an empty list. These return codes indicate the success,
            failure, or other status of the processor object.
        use_safe_process : bool, optional
            Flag indicating whether to use safe process, by default True.
            Safe process catches errors and logs them, rather than crashing.
        """
        self.io_manager = io_manager
        self.image_operator = image_operator
        self.log_keys = log_keys
        self.save_return_codes = save_return_codes
        self.use_safe_process = use_safe_process

    def fit(self, batch_processor: "BatchProcessor") -> "Processor":
        '''Copy over fit values from the batch processor.

        This method copies over fit values from the given batch processor to
        the current instance.
        It iterates through all attributes of the batch processor and checks if
        the attribute name ends with an underscore, indicating that it is a
        fit variable. If so, it copies the attribute value to the current instance.

        Parameters
        ----------
        batch_processor : BatchProcessor
            The batch processor from which to copy the fit values.

        Returns
        -------
        self : Processor
            The current instance with the fit values copied over.
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
        '''Process one image, with metadata stored in row and other context in
        resources.

        Generally speaking, src refers to our new data, and dst refers to
        the existing data (including if the existing data was just updated
        with src in a previous row).

        A better name than "process_row" may be available. process_row is not
        100% desirable since we don't clarify what a row is..

        Parameters
        ----------
        i : int
            The index of the image being processed.
        row : pd.Series
            The metadata of the image being processed, including the filename.
        resources : dict
            Additional context and resources for the image processing.

        Returns
        -------
        pd.Series
            The updated metadata of the image after processing.
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

    def safe_process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        '''Same as process, but catching anticipated errors.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        src : np.ndarray
            Source image.
        dst : np.ndarray
            Destination image.

        Returns
        -------
        results : dict
            Dictionary containing the results of the processing.
            Possible keys:
            - blended_img: Combined image. Not always returned.
            - M: Homography transform. Not always returned.
            - src_kp: Keypoints for the src image. Not always returned.
            - src_des: KP descriptors for the src image. Not always returned.
            - duration: Time spent.
            - return_code: Code indicating the outcome of the processing.
                Possible values:
                - 'success': Processing completed successfully.
                - 'opencv_err': OpenCV error occurred.
                - 'bad_det': Homography transform error occurred.
                - 'dark_frame': Src dark frame error occurred.
                - 'dst_dark_frame': Dst dark frame error occurred.
                - 'linalg_err': Linear algebra error occurred.
                - 'out_of_bounds': Out of bounds error occurred.
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
        except exceptions.OutOfBoundsError:
            return_code = 'out_of_bounds'

        duration = time.time() - start
        results['duration'] = duration
        results['return_code'] = return_code

        return results

    @abstractmethod
    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:
        '''Abstract base method for getting the source image.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.

        Returns
        -------
        dict
            Results dictionary.
        '''

    @abstractmethod
    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:
        '''Abstract base method for getting the destination image.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.

        Returns
        -------
        dict
            Results dictionary.
        '''

    @abstractmethod
    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        '''Abstract base method for main processing function.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        src:
            Dictionary containing source image and parameters.
        dst:
            Dictionary containing destination image and parameters.

        Returns
        -------
        dict
            Results dictionary.
        '''

    @abstractmethod
    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ) -> pd.Series:
        '''Abstract base method for storing results.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        results:
            Dictionary containing output of processing.

        Returns
        -------
        pd.Series
            Metadata for the processed image.
        '''


class DatasetProcessor(Processor):
    '''Class for operating on src images and a dataset from which we draw dst images.
    Primary role is to be a parent class that enables shared functionality between
    DatasetUpdater and DatasetScorer.
    '''

    def __init__(
        self,
        io_manager: IOManager,
        image_operator: BaseImageOperator,
        log_keys: list[str] = [],
        save_return_codes: list[str] = [],
        use_safe_process: bool = True,
        dtype: str = 'uint8',
    ):
        """
        Initialize a Dataset Processor object.

        Parameters
        ----------
        io_manager : IOManager
            The IOManager object responsible for managing input/output operations.
        image_operator : BaseImageOperator
            The image operator object used for the operations at the core of
            processing.
        log_keys : list[str], optional
            The list of local variables to log in tabular form,
            by default an empty list.
        save_return_codes : list[str], optional
            The list of return codes to save to the progress images directory,
            by default an empty list. These return codes indicate the success,
            failure, or other status of the processor object.
        use_safe_process : bool, optional
            Flag indicating whether to use safe process, by default True.
            Safe process catches errors and logs them, rather than crashing.
        dtype : str
            Data type to use for the image, by default 'uint8'.
        """

        super().__init__(
            io_manager=io_manager,
            image_operator=image_operator,
            log_keys=log_keys,
            save_return_codes=save_return_codes,
            use_safe_process=use_safe_process,
        )
        self.dtype = getattr(np, dtype)

    def get_src(self, i: int, row: pd.Series, resources: dict) -> dict:
        '''Load an image using the "filepath" column of the metadata.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.

        Returns
        -------
        dict
            Results dictionary.
        '''

        LOGGER.info('Getting src...')

        src_img = ImageIO.load(
            row['filepath'],
            dtype=self.dtype,
        )

        return {'image': src_img}

    def get_dst(self, i: int, row: pd.Series, resources: dict) -> dict:
        '''Load an image from the dataset stored in resources, using the row's
        x_off, y_off, x_size, and y_size.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.

        Returns
        -------
        dict
            Results dictionary.
        '''

        LOGGER.info('Getting dst...')

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

    def get_image_from_dataset(
        self,
        dataset: gdal.Dataset,
        x_off: int,
        y_off: int,
        x_size: int,
        y_size: int,
    ) -> np.ndarray:
        '''Get an image from the dataset.

        This method retrieves an image from the given dataset based on the specified
        offset and size.

        Parameters
        ----------
        dataset : gdal.Dataset
            The GDAL dataset from which to retrieve the image.
        x_off : int
            The x-coordinate offset of the image within the dataset.
        y_off : int
            The y-coordinate offset of the image within the dataset.
        x_size : int
            The width of the image.
        y_size : int
            The height of the image.

        Returns
        -------
        numpy.ndarray
            The image as a NumPy array.

        Raises
        ------
        AssertionError
            If the specified offsets and sizes are out of bounds of the dataset.

        '''
        assert x_off >= 0, 'x_off cannot be less than 0'
        assert x_off + x_size <= dataset.RasterXSize, \
            'x_off + x_size cannot be greater than self.x_size_'
        assert y_off >= 0, 'y_off cannot be less than 0'
        assert y_off + y_size <= dataset.RasterYSize, \
            'y_off + y_size cannot be greater than self.y_size_'

        # Note that we cast the input as int, in case the input was numpy
        # integers instead of Python integers.
        img = dataset.ReadAsArray(
            xoff=int(x_off),
            yoff=int(y_off),
            xsize=int(x_size),
            ysize=int(y_size),
        )
        img = img.transpose(1, 2, 0)

        self.update_log(locals())

        return img

    def save_image_to_dataset(
        self,
        dataset: gdal.Dataset,
        img: np.ndarray,
        x_off: int,
        y_off: int,
    ):
        '''Save an image to a gdal Dataset.

        Parameters
        ----------
        dataset : gdal.Dataset
            The GDAL dataset from which to retrieve the image.
        img : np.ndarray
            The image to save.
        x_off : int
            The x-coordinate offset of the image within the dataset.
        y_off : int
            The y-coordinate offset of the image within the dataset.
        '''

        img_to_save = img.transpose(2, 0, 1)
        dataset.WriteArray(
            img_to_save,
            xoff=int(x_off),
            yoff=int(y_off),
        )

        self.update_log(locals())


class DatasetUpdater(DatasetProcessor):
    '''Class for updating a dataset based on the provided image.
    '''

    def process(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        src: dict,
        dst: dict,
    ) -> dict:
        '''Processing for the dataset updater.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        src:
            Dictionary containing source image and parameters.
        dst:
            Dictionary containing destination image and parameters.

        Returns
        -------
        dict
            Results dictionary.
        '''

        LOGGER.info('Performing image operation...')

        # Combine the images
        results = self.image_operator.operate(
            src['image'],
            dst['image'],
        )
        self.update_log(self.image_operator.log)

        results['src_image'] = src['image']
        results['dst_image'] = dst['image']

        return results

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ) -> pd.Series:
        '''Store results, i.e. update the dataset.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        results:
            Results dictionary to store.
        '''

        LOGGER.info('Updating dataset...')

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
        if 'progress_images_dir' in self.io_manager.output_filepaths:
            progress_images_dir = (
                self.io_manager.output_filepaths['progress_images_dir']
            )

            if (
                (progress_images_dir is not None)
                and (results['return_code'] in self.save_return_codes)
            ):

                # Get the images to save
                src_img = ImageIO.load(
                    row['filepath'],
                    dtype=self.dtype,
                )
                dst_img = self.get_image_from_dataset(
                    resources['dataset'],
                    row['x_off'],
                    row['y_off'],
                    row['x_size'],
                    row['y_size'],
                )

                # Make a progress images dir
                os.makedirs(progress_images_dir, exist_ok=True)

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

                cv2.imwrite(src_fp, src_img[:, :, ::-1])
                cv2.imwrite(dst_fp, dst_img[:, :, ::-1])

                if 'blended_img' in results:
                    blended_fp = os.path.join(
                        progress_images_dir,
                        f'{n_tests_existing:06d}_blended.tiff'
                    )
                    cv2.imwrite(blended_fp, results['blended_img'][:, :, ::-1])

        return row


class ReferencerDatasetUpdater(DatasetUpdater):
    '''Class for not just updating the dataset, but also saving now-referenced
    images based on the processing done.
    '''

    def store_results(
        self,
        i: int,
        row: pd.Series,
        resources: dict,
        results: dict,
    ):
        '''Store results, i.e. update the dataset and save a referenced image.

        Parameters
        ----------
        i : int
            Index of the current image.
        row : pd.Series
            Series containing the metadata for the current image.
        resources : dict
            Dictionary containing additional resources.
        results:
            Results dictionary to store.
        '''

        LOGGER.info('Starting DatasetRegistrar.store_results...')

        # Update the dataset
        row = super().store_results(i, row, resources, results)

        # Store the image
        if results['return_code'] == 'success':

            LOGGER.info('Saving image...')

            transformer: RasterCoordinateTransformer = resources['transformer']

            # Get filepath
            fp_pattern = self.io_manager.output_filepaths['referenced_images']
            fp = fp_pattern.format(row.name)

            # Get the bounds in physical coordinates
            (
                x_off_dstframe, y_off_dstframe,
                x_size, y_size,
            ) = [int(np.round(_)) for _ in results['warped_bounds']]
            x_off = row['x_off'] + x_off_dstframe
            y_off = row['y_off'] + y_off_dstframe
            x_min, x_max, y_min, y_max = transformer.pixel_to_physical(
                x_off, y_off, x_size, y_size
            )

            # Get the (trimmed) image
            warped_image = results['warped_image'][
                y_off_dstframe:y_off_dstframe + y_size,
                x_off_dstframe:x_off_dstframe + x_size,
            ]

            # Save the referenced image
            ReferencedImageIO.save(
                filepath=fp,
                img=warped_image,
                x_bounds=[x_min, x_max],
                y_bounds=[y_min, y_max],
                crs=pyproj.CRS(resources['dataset'].GetProjection()),
            )
            row['output_filepath'] = fp

            # Update the row values
            row[['x_off', 'y_off', 'x_size', 'y_size']] = [
                x_off, y_off, x_size, y_size
            ]

        return row
