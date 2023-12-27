import numpy as np
import pandas as pd
import scipy
# This is a draft---don't overengineer!
# NO renaming!
# NO refactoring!
# TODO: Remove this when the draft is done.

from abc import abstractmethod
import inspect
import os
import pickle
import re
import shutil

from osgeo import gdal
import yaml


class FileManager:

    def __init__(
        self,
        out_dir: str,
        filename: str,
        file_exists: str = 'error',
        aux_files: dict[str] = {},
        checkpoint_freq: int = 100,
        checkpoint_subdir: str = 'checkpoints',
    ):

        self.out_dir = out_dir
        self.filename = filename
        self.file_exists = file_exists
        self.aux_files = aux_files
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_subdir = checkpoint_subdir

    # TODO: Maybe remove trailing underscores since they're not directly
    # attrs of a fit estimator?
    def prepare_filetree(self):

        # Main filepath parameters
        self.out_dir_ = self.out_dir
        self.filepath_ = os.path.join(self.out_dir_, self.filename)

        # Flexible file-handling. TODO: Maybe overkill?
        if os.path.isfile(self.filepath_):

            # Standard, simple options
            if self.file_exists == 'error':
                raise FileExistsError('File already exists at destination.')
            elif self.file_exists in ['pass', 'load']:
                pass
            elif self.file_exists == 'overwrite':
                shutil.rmtree(self.out_dir_)
            # Create a new file with a new number appended
            elif self.file_exists == 'new':
                out_dir_pattern = self.out_dir_ + '_v{:03d}'
                i = 0
                while os.path.isfile(self.filepath_):
                    self.out_dir_ = out_dir_pattern.format(i)
                    self.filepath_ = os.path.join(self.out_dir_, self.filename)
                    i += 1
            else:
                raise ValueError(
                    'Unrecognized value for filepath, '
                    f'filepath={self.filepath_}'
                )

        # Checkpoints file handling
        self.checkpoint_subdir_ = os.path.join(
            self.out_dir_, self.checkpoint_subdir)
        base, ext = os.path.splitext(self.filename)
        i_tag = '_i{:06d}'
        self.checkpoint_filepattern_ = base + i_tag + ext

        # Auxiliary files
        self.aux_filepaths_ = {}
        for key, filename in self.aux_files.items():
            self.aux_filepaths_[key] = os.path.join(self.out_dir_, filename)

        # Ensure directories exist
        os.makedirs(self.out_dir_, exist_ok=True)
        os.makedirs(self.checkpoint_subdir_, exist_ok=True)

        return self.out_dir_, self.filepath_

    def save_settings(self, obj):

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
        with open(self.aux_filepaths_['settings'], 'w') as file:
            yaml.dump(settings, file)

    def search_for_checkpoint(self):

        # Look for checkpoint files
        i_start = -1
        filename = None
        search_pattern = self.checkpoint_filepattern_.replace(
            r'{:06d}',
            '(\\d{6})\\',
        )
        pattern = re.compile(search_pattern)
        possible_files = os.listdir(self.checkpoint_subdir_)
        for j, filename in enumerate(possible_files):
            match = pattern.search(filename)
            if not match:
                continue

            number = int(match.group(1))
            if number > i_start:
                i_start = number
                filename = possible_files[j]

        # We don't want to start on the same loop that was saved, but the
        # one after
        i_start += 1

        return i_start, filename

    @abstractmethod
    def save_to_checkpoint(self, i):
        pass

    @abstractmethod
    def load_from_checkpoint(self, i, filename):
        pass

    def search_and_load_checkpoint(self):

        i_resume, checkpoint_filename = self.search_for_checkpoint()
        loaded_data = self.load_from_checkpoint(i_resume, checkpoint_filename)

        return i_resume, loaded_data


class MosaicFileManager(FileManager):

    def open_dataset(self):

        return gdal.Open(self.filepath_, gdal.GA_Update)

    def save_to_checkpoint(self, i, dataset, y_pred=None):

        # Conditions for normal return
        if self.checkpoint_freq is None:
            return dataset
        if (i % self.checkpoint_freq != 0) or (i == 0):
            return dataset

        # Flush data to disk
        dataset.FlushCache()
        dataset = None

        # Make checkpoint file by copying the dataset
        checkpoint_fp = os.path.join(
            self.checkpoint_subdir_,
            self.checkpoint_filepattern_.format(i),
        )
        shutil.copy(self.filepath_, checkpoint_fp)

        # Store auxiliary files
        if y_pred is not None:
            y_pred.to_csv(self.aux_filepaths_['y_pred'])

        # Re-open dataset
        dataset = self.open_dataset()

        return dataset

    def load_from_checkpoint(self, i_checkpoint, checkpoint_filename):

        if i_checkpoint == 0:
            return None

        print(f'Loading checkpoint file for i={i_checkpoint}')

        # Copy checkpoint dataset
        filepath = os.path.join(self.checkpoint_subdir_, checkpoint_filename)
        shutil.copy(filepath, self.filepath_)

        # And load the predictions
        y_pred = pd.read_csv(self.aux_filepaths_['y_pred'], index_col=0)

        loaded_data = {
            'y_pred': y_pred,
        }
        return loaded_data
