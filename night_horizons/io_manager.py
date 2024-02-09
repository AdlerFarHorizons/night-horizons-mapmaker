from abc import abstractmethod
import copy
import glob
import inspect
import os
import pickle
import re
import shutil
from typing import Tuple, Union

from osgeo import gdal
gdal.UseExceptions()
import yaml

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from night_horizons.data_io import GDALDatasetIO


class IOManager:
    '''
    Things IOManager should do:

    - Get full paths given relative paths
    - Identify all valid files in a directory and subdirectories
    - Check if a file exists, and act accordingly (overwrite, etc.) - DONE
    - Read files
    - Save files
    - Checkpoint files - DONE
    - Save auxiliary files (settings, logs, etc.) - DONE (kinda)

    TODO: There should be a way to specify what files a given class needs
        and how that's hooked up to IOManager.

    NOTE: This *could* be broken into an InputFileManager, an
    OutputFileManager, and a DataIOManager, but that seems like overkill.

    Parameters
    ----------
    Returns
    -------
    '''

    def __init__(
        self,
        input_dir: str,
        input_description: dict[dict],
        output_dir: str,
        output_description: dict[str],
        root_dir: str = None,
        file_exists: str = 'error',
        tracked_file_key: str = None,
        checkpoint_subdir: str = 'checkpoints',
        checkpoint_selection: list[str] = None,
        checkpoint_tag: str = '_i{:06d}',
        checkpoint_freq: int = 100,
        data_ios: dict[str] = {},
    ) -> None:

        if root_dir is not None:
            input_dir = os.path.join(root_dir, input_dir)
            output_dir = os.path.join(root_dir, output_dir)

        if checkpoint_selection is None:
            checkpoint_selection = list(output_description.keys())

        self.input_dir = input_dir
        self.output_description = output_description
        self.root_dir = root_dir
        self.file_exists = file_exists
        self.tracked_file_key = tracked_file_key
        self.checkpoint_subdir = checkpoint_subdir
        self.checkpoint_selection = checkpoint_selection
        self.checkpoint_tag = checkpoint_tag
        self.checkpoint_freq = checkpoint_freq
        self.data_ios = data_ios

        # Process input filetree
        self.input_filepaths, self.input_description = \
            self.find_input_files(input_description)

        # Process output filetree
        # TODO: Ideally this would be called at the time of the fit.
        self.output_filepaths, self.output_dir = \
            self.get_output_filepaths(
                output_dir=output_dir,
                output_description=output_description,
                file_exists=file_exists,
                tracked_file_key=tracked_file_key,
            )

        # And finally, the checkpoint info
        self.checkpoint_filepatterns, self.checkpoint_dir = \
            self.get_checkpoint_filepatterns(
                output_dir=self.output_dir,
                output_filepaths=self.output_filepaths,
                checkpoint_subdir=self.checkpoint_subdir,
                checkpoint_selection=self.checkpoint_selection,
                checkpoint_tag=self.checkpoint_tag,
            )

    def find_input_files(
        self,
        input_description: dict[dict],
    ) -> Tuple[dict[pd.Series], dict[dict]]:

        # Validate and store input description
        modified_input_description = copy.deepcopy(input_description)
        for key, descr in modified_input_description.items():
            if isinstance(descr, str):
                modified_input_description[key] = \
                    os.path.join(self.input_dir, descr)
            else:
                if 'directory' not in descr:
                    raise ValueError(
                        f'input_description[{key}] must have a "directory" '
                        'key if it is a dictionary'
                    )
                modified_input_description[key]['directory'] = \
                    os.path.join(self.input_dir, descr['directory'])

        # Find files
        input_filepaths = {
            key: (
                self.find_files(**item)
                if isinstance(item, dict)
                else item
            )
            for key, item in modified_input_description.items()
        }

        return input_filepaths, modified_input_description

    def find_files(
        self,
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

    def get_output_filepaths(
        self,
        output_dir: str,
        output_description: dict[str],
        file_exists: str,
        tracked_file_key: str,
    ) -> Tuple[dict[str], str]:

        # Exit early if there's nothing to do
        if len(output_description) == 0:
            return {}, output_dir

        # Default to the first key
        if tracked_file_key is None:
            tracked_file_key = list(output_description.keys())[0]
        tracked_filename = output_description[tracked_file_key]

        # Main filepath parameters
        tracked_filepath = os.path.join(output_dir, tracked_filename)
        if os.path.isfile(tracked_filepath):

            # Standard, simple options
            if file_exists == 'error':
                raise FileExistsError('File already exists at destination.')
            elif file_exists in ['pass', 'load']:
                pass
            elif file_exists == 'overwrite':
                shutil.rmtree(output_dir)
            # Create a new file with a new number appended
            elif file_exists == 'new':
                out_dir_pattern = output_dir + '_v{:03d}'
                i = 0
                while os.path.isfile(tracked_filepath):
                    output_dir = out_dir_pattern.format(i)
                    tracked_filepath = os.path.join(
                        output_dir, tracked_filename)
                    i += 1
            else:
                raise ValueError(
                    'Unrecognized value for filepath, '
                    f'filepath={tracked_filepath}'
                )

        # Auxiliary files
        output_filepaths = {}
        for key, tracked_filename in output_description.items():
            output_filepaths[key] = os.path.join(
                output_dir,
                tracked_filename,
            )

        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)

        return output_filepaths, output_dir

    def save_settings(self, obj):
        '''TODO: Another thing to move into a DataIO

        Parameters
        ----------
        Returns
        -------
        '''

        fullargspec = inspect.getfullargspec(type(obj))
        settings = {}
        for setting in fullargspec.args:
            if setting == 'self':
                continue
            value = getattr(obj, setting)
            try:
                pickle.dumps(value)
            except TypeError:
                value = 'no string repr'
            settings[setting] = value
        with open(self.output_filepaths['settings'], 'w') as file:
            yaml.dump(settings, file)

    def get_checkpoint_filepatterns(
        self,
        output_dir: str,
        output_filepaths: dict[str],
        checkpoint_subdir: str,
        checkpoint_selection: list[str],
        checkpoint_tag: str,
    ) -> Tuple[dict[str], str]:

        checkpoint_dir = os.path.join(output_dir, checkpoint_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint filepatterns
        checkpoint_filepatterns = {}
        for key in checkpoint_selection:
            filepath = output_filepaths[key]
            base, ext = os.path.splitext(os.path.basename(filepath))
            checkpoint_filepatterns[key] = base + checkpoint_tag + ext

        return checkpoint_filepatterns, checkpoint_dir

    def search_for_checkpoint(self, key: str = None):

        if key is None:
            key = self.tracked_file_key

        checkpoint_filepattern = self.checkpoint_filepatterns[key]

        # Look for checkpoint files
        i_resume = -1
        filename = None
        search_pattern = checkpoint_filepattern.replace(
            r'{:06d}',
            '(\\d{6})\\',
        )
        pattern = re.compile(search_pattern)
        possible_files = os.listdir(self.checkpoint_dir)
        filename_start = None
        for j, filename in enumerate(possible_files):
            match = pattern.search(filename)
            if not match:
                continue

            number = int(match.group(1))
            if number > i_resume:
                i_resume = number
                filename_start = possible_files[j]

        # We don't want to start on the same loop that was saved, but the
        # one after
        i_resume += 1

        return i_resume

    @abstractmethod
    def save_to_checkpoint(self, i):
        pass

    @abstractmethod
    def load_from_checkpoint(self, i, filename):
        pass

    def search_and_load_checkpoint(self, key: str = None):

        i_resume = self.search_for_checkpoint(key=key)
        loaded_data = self.load_from_checkpoint(i_resume)

        return i_resume, loaded_data


class MosaicIOManager(IOManager):

    def __init__(
        self,
        input_dir: str,
        input_description: dict[dict],
        output_dir: str,
        output_description: dict[str] = {
            'mosaic': 'mosaic.tiff',
            'settings': 'settings.yaml',
            'log': 'log.csv',
            'y_pred': 'y_pred.csv',
            'progress_images_dir': 'progress_images',
            'referenced_images': 'referenced_images/img_ind{:06d}.tiff',
        },
        root_dir: str = None,
        file_exists: str = 'error',
        tracked_file_key: str = 'mosaic',
        checkpoint_subdir: str = 'checkpoints',
        checkpoint_selection: list[str] = [
            'mosaic', 'settings', 'log', 'y_pred'],
        checkpoint_tag: str = '_i{:06d}',
        checkpoint_freq: int = 100,
    ):
        '''The inputs are more-appropriate defaults for mosaics.

        TODO: Consider renaming referenced_images to georeferenced_images
        everywhere.

        Parameters
        ----------
        Returns
        -------
        '''

        super().__init__(
            input_dir=input_dir,
            input_description=input_description,
            output_dir=output_dir,
            output_description=output_description,
            root_dir=root_dir,
            file_exists=file_exists,
            tracked_file_key=tracked_file_key,
            checkpoint_subdir=checkpoint_subdir,
            checkpoint_selection=checkpoint_selection,
            checkpoint_tag=checkpoint_tag,
            checkpoint_freq=checkpoint_freq,
        )

    def open_dataset(self):
        '''
        TODO: Kind of awkard that this is one of the only convenience functions
        for opening/loading data.
        '''

        return GDALDatasetIO.load(
            self.output_filepaths['mosaic'],
            mode=gdal.GA_Update,
        )

    def save_to_checkpoint(self, i, dataset, y_pred=None):

        # Conditions for normal return
        if self.checkpoint_freq is None:
            return dataset
        if (i % self.checkpoint_freq != 0) or (i == 0):
            return dataset

        # Flush data to disk
        dataset.FlushCache()
        dataset = None

        # Store auxiliary files
        if y_pred is not None:
            y_pred.to_csv(self.output_filepaths['y_pred'])

        # Make checkpoint files by copying the data
        for key, pattern in self.checkpoint_filepatterns.items():
            if os.path.isfile(self.output_filepaths[key]):
                checkpoint_fp = os.path.join(
                    self.checkpoint_dir,
                    pattern.format(i)
                )
                shutil.copy(self.output_filepaths[key], checkpoint_fp)

        # Re-open dataset
        dataset = self.open_dataset()

        return dataset

    def load_from_checkpoint(self, i_checkpoint):

        if i_checkpoint == 0:
            return None

        print(f'Loading checkpoint file for i={i_checkpoint}')

        # Copy checkpoint files
        for key, pattern in self.checkpoint_filepatterns.items():
            checkpoint_fp = os.path.join(
                self.checkpoint_dir,
                pattern.format(i_checkpoint - 1)
            )
            if os.path.isfile(checkpoint_fp):
                shutil.copy(checkpoint_fp, self.output_filepaths[key])

        # And load the predictions
        y_pred = pd.read_csv(self.output_filepaths['y_pred'], index_col=0)

        loaded_data = {
            'y_pred': y_pred,
        }
        return loaded_data


class TrainMosaicIOManager(MosaicIOManager):

    def __init__(
        self,
        output_description: dict = {
            'mosaic': 'mosaic.tiff',
            'settings': 'settings_train.yaml',
            'log': 'log_train.yaml',
            'y_pred': 'y_pred_train.csv',
            'progress_images_dir_train': 'progress_images_train',
        },
        file_exists: str = 'pass',
        *args, **kwargs
    ):
        '''The inputs are those suited for a training mosaic.
        See MosaicIOManager for all keyword arguments.

        Parameters
        ----------
        Returns
        -------
        '''

        super().__init__(
            output_description=output_description,
            file_exists=file_exists,
            *args, **kwargs
        )
