from abc import abstractmethod
import copy
import inspect
import os
import pickle
import re
import shutil
from typing import Tuple, Union

import numpy as np
from osgeo import gdal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine as SQLEngine
import yaml

from night_horizons.data_io import GDALDatasetIO

gdal.UseExceptions()


class IOManager:
    '''Class for managing input and output files.
    Some responsibilities:

    - Get full paths given relative paths
    - Identify all valid files in a directory and subdirectories
    - Check if a file exists, and act accordingly (overwrite, etc.)
    - Checkpoint files
    '''

    def __init__(
        self,
        input_dir: str,
        input_description: dict[dict],
        output_dir: str,
        output_description: dict[str],
        root_dir: str = None,
        file_exists: str = 'error',
        checkpoint_file_key: str = None,
        checkpoint_subdir: str = 'checkpoints',
        checkpoint_selection: list[str] = None,
        checkpoint_tag: str = '_i{:06d}',
        checkpoint_freq: int = 100,
    ):
        """Initialize the IOManager object.

        Parameters
        ----------
        input_dir : str
            The directory containing the input files.
        input_description : dict[dict]
            The description of the input files.
        output_dir : str
            The directory where the output files will be saved.
        output_description : dict[str]
            The description of the output files.
        root_dir : str, optional
            The root directory. If provided, the input_dir and output_dir will
            be relative to this directory.
        file_exists : str, optional
            The action to take if a file already exists. Defaults to 'error'.
            Other options are 'pass', 'load', 'overwrite', and 'new'.
        checkpoint_file_key : str, optional
            The key for the file that determines what checkpoint files to use.
        checkpoint_subdir : str, optional
            The subdirectory where checkpoints will be saved.
        checkpoint_selection : list[str], optional
            The list of output files to use for checkpointing.
        checkpoint_tag : str, optional
            The tag to append to checkpoint filenames. Defaults to '_i{:06d}'.
        checkpoint_freq : int, optional
            The frequency at which to save checkpoints. Defaults to 100.
        data_ios : dict[str], optional
            The dictionary of additional data IOs. Defaults to {}.
        """
        if root_dir is not None:
            input_dir = os.path.join(root_dir, input_dir)
            output_dir = os.path.join(root_dir, output_dir)

        if checkpoint_selection is None:
            checkpoint_selection = list(output_description.keys())

        self.input_dir = input_dir
        self.output_description = output_description
        self.root_dir = root_dir
        self.file_exists = file_exists
        self.checkpoint_file_key = checkpoint_file_key
        self.checkpoint_subdir = checkpoint_subdir
        self.checkpoint_selection = checkpoint_selection
        self.checkpoint_tag = checkpoint_tag
        self.checkpoint_freq = checkpoint_freq

        # Process input filetree
        self.input_filepaths, self.input_description = \
            self.find_input_files(input_description)

        # Process output filetree
        self.output_filepaths, self.output_dir = \
            self.get_output_filepaths(
                output_dir=output_dir,
                output_description=output_description,
                file_exists=file_exists,
                tracked_file_key=checkpoint_file_key,
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
        '''Find input files based on the provided input description.

        Parameters
        ----------
        input_description : dict[dict]
            A dictionary containing the input description. Each key represents
            a specific input file, and the corresponding value is a dictionary
            describing the file. If the value is a string, it is treated as a
            file path relative to the input directory. If the value is a
            dictionary, it must contain arguments to select files based on
            find_selected_files.

        Returns
        -------
        Tuple[dict[pd.Series], dict[dict]]
            A tuple containing two dictionaries. The first dictionary maps each
            input file key to a pandas Series object representing the selected
            files. The second dictionary is a modified version of the input
            description, where file paths have been resolved relative to the
            input directory.
        '''

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
                self.find_selected_files(**item)
                if isinstance(item, dict)
                else item
            )
            for key, item in modified_input_description.items()
        }

        return input_filepaths, modified_input_description

    def find_selected_files(
        self,
        directory: str,
        extension: str = None,
        pattern: str = None,
    ) -> pd.Series:
        '''
        Parameters
        ----------
        directory :
            Directory containing the data.
        extension :
            What filetypes to include.

        Returns
        -------
        fps :
            Data filepaths.
        '''

        fps = self.find_files(directory)

        fps = self.select_files(fps, extension, pattern)

        return fps

    def find_files(self, directory: str) -> pd.Series:
        """Find all files in the specified directory and its subdirectories.

        Parameters
        ----------
        directory : str
            The directory to search for files.

        Returns
        -------
        pd.Series
            A pandas Series containing the file paths of all the found files.

        Raises
        ------
        AssertionError
            If the specified directory is not a valid directory.
        """
        assert os.path.isdir(directory), f'{directory} is not a directory.'

        # Walk the tree to get files
        fps = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                fps.append(os.path.join(root, name))
        fps = pd.Series(fps)

        return fps

    def select_files(
        self,
        fps: pd.Series,
        extension: Union[str, list] = None,
        pattern: str = None
    ):
        """Selects files from a pandas Series based on the given
        extension and pattern.

        Parameters
        ----------
        fps : pd.Series
            The pandas Series containing file paths.
        extension : Union[str, list], optional
            The file extension(s) to filter by. Defaults to None.
        pattern : str, optional
            The pattern to match against the file paths. Defaults to None.

        Returns
        -------
        pd.Series
            The filtered pandas Series containing selected file paths.
        """

        # Handle extensions.
        if extension is not None:
            if pattern is None:
                pattern = '.*'
            else:
                pattern += '.*'
            # When a single extension
            if isinstance(extension, str):
                pattern += f'{extension}$'
            # When a list of extensions
            else:
                pattern += '(' + '|'.join(extension) + ')$'

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
        '''
        Parameters
        ----------
        output_dir : str
            The directory where the output files will be saved.
        output_description : dict[str]
            The description of the output files.
        file_exists : str
            The action to take if a file already exists. Options include
            'error', 'pass', 'load', 'overwrite', and 'new'.
        tracked_file_key : str
            The key for the file that determines what checkpoint files to use.

        Returns
        -------
        Tuple[dict[str], str]
            A tuple containing two elements. The first element is a dictionary
            mapping each output file key to the corresponding file path. The
            second element is the output directory.
        '''

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
        '''Save the settings of an object to a file.
        
        This may be degenerate with saving the config, but better safe than
        sorry, and it's computationally inexpensive.

        Parameters
        ----------
        obj : object
            The object to save the settings of.
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
        '''Get the checkpoint file patterns for saving checkpoints.

        Parameters
        ----------
        output_dir : str
            The output directory where the checkpoints will be saved.
        output_filepaths : dict[str]
            A dictionary mapping output keys to their corresponding filepaths.
        checkpoint_subdir : str
            The subdirectory within the output directory where
            the checkpoints will be saved.
        checkpoint_selection : list[str]
            A list of keys from the output_filepaths dictionary to select
            for creating checkpoint file patterns.
        checkpoint_tag : str
            The tag to be added to the base filename of each checkpoint file.

        Returns
        -------
        Tuple[dict[str], str]
            A tuple containing the checkpoint file patterns and
            the checkpoint directory.
        '''

        checkpoint_dir = os.path.join(output_dir, checkpoint_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint filepatterns
        checkpoint_filepatterns = {}
        for key in checkpoint_selection:
            filepath = output_filepaths[key]
            base, ext = os.path.splitext(os.path.basename(filepath))
            checkpoint_filepatterns[key] = base + checkpoint_tag + ext

        return checkpoint_filepatterns, checkpoint_dir

    def search_for_checkpoint(
        self,
        key: str = None,
    ) -> Tuple[int, list[str]]:
        '''Search for checkpoint files in the specified directory.

        Parameters
        ----------
        key : str, optional
            The key to identify the checkpoint file pattern. If not provided,
            the default key will be used.

        Returns
        -------
        Tuple[int, list[str]]
            A tuple containing the latest checkpoint index and a list of
            filenames matching the checkpoint file pattern.
        '''

        if key is None:
            key = self.checkpoint_file_key

        checkpoint_filepattern = self.checkpoint_filepatterns[key]

        # Look for checkpoint files
        i_latest = -1
        filename = None
        search_pattern = checkpoint_filepattern.replace(
            r'{:06d}',
            '(\\d{6})\\',
        )
        pattern = re.compile(search_pattern)
        possible_files = os.listdir(self.checkpoint_dir)
        filenames = []
        for j, filename in enumerate(possible_files):
            match = pattern.search(filename)
            if not match:
                continue

            number = int(match.group(1))
            filenames.append((number, filename))
            if number > i_latest:
                i_latest = number

        return i_latest, filenames

    @abstractmethod
    def save_to_checkpoint(self, i: int):
        '''Save the current state of the object to a checkpoint file.

        Parameters
        ----------
        i : int
            The index of the checkpoint.
        '''

    @abstractmethod
    def load_from_checkpoint(self, i: int):
        '''Load the state of the object from a checkpoint file.

        Parameters
        ----------
        i : int
            The index of the checkpoint.
        '''

    def search_and_load_checkpoint(
        self,
        key: str = None
    ) -> Tuple[int, object]:
        '''
        Searches for the latest checkpoint with the given key and loads
        the data from the next checkpoint.

        Parameters
        ----------
        key : str, optional
            The key to search for in the checkpoints. If not provided,
            searches for the latest checkpoint.

        Returns
        -------
        tuple
            A tuple containing the index of the next checkpoint and the
            loaded data from that checkpoint.
        '''

        i_latest, _ = self.search_for_checkpoint(key=key)

        # We don't want to start on the same loop that was saved, but the
        # one after
        i_resume = i_latest + 1
        loaded_data = self.load_from_checkpoint(i_resume)

        return i_resume, loaded_data

    def prune_checkpoints(self):
        """Prunes the checkpoint files in the checkpoint directory,
        keeping only the latest ones.

        This method finds the latest checkpoint files based on the patterns
        specified in `checkpoint_filepatterns`.
        It then deletes all the files in the checkpoint directory
        that are not the latest ones.
        """
        for key in self.checkpoint_filepatterns.keys():
            # Find the latest files
            i_latest, filenames = self.search_for_checkpoint(key=key)

            # Delete all but the latest files
            for (i_file, filename) in filenames:
                if i_file != i_latest:
                    os.remove(os.path.join(self.checkpoint_dir, filename))

    def get_connection(self, url: str = None) -> SQLEngine:
        '''Returns a database connection.

        Parameters
        ----------
        url : str, optional
            The URL of the database connection. If not provided, it will use
            the value of the 'DATABASE_URL' environment variable.

        Returns
        -------
        engine : sqlalchemy.engine.Engine
            The database connection engine.

        '''

        if url is None:
            url = os.getenv('DATABASE_URL')
            url = url.replace('postgres://', 'postgresql+pyscopg2://')
        engine = create_engine(url)

        return engine


class MosaicIOManager(IOManager):
    '''Class for managing input and output files for mosaics.
    '''

    def __init__(
        self,
        input_dir: str,
        input_description: dict[dict],
        output_dir: str,
        output_description: dict[str] = {
            'mosaic': 'mosaic.tiff',
            'settings': 'settings.yaml',
            'log': 'log.csv',
        },
        root_dir: str = None,
        file_exists: str = 'error',
        tracked_file_key: str = 'mosaic',
        checkpoint_subdir: str = 'checkpoints',
        checkpoint_selection: list[str] = ['mosaic', 'settings', 'log'],
        checkpoint_tag: str = '_i{:06d}',
        checkpoint_freq: int = 100,
    ):
        '''Initialize the IOManager object.

        This method initializes the IOManager object with the provided parameters.

        Parameters
        ----------
        input_dir : str
            The directory path where the input files are located.

        input_description : dict[dict]
            A dictionary containing the description of the input files.

        output_dir : str
            The directory path where the output files will be saved.

        output_description : dict[str], optional
            A dictionary containing the description of the output files.
            The default value is:
            {
                'mosaic': 'mosaic.tiff',
                'settings': 'settings.yaml',
                'log': 'log.csv',
            }

        root_dir : str, optional
            The root directory path. If provided, it will be used as the base directory for all file paths.
            The default value is None.

        file_exists : str, optional
            The action to take if a file already exists.
            Possible values are 'error', 'overwrite', 'skip'.
            The default value is 'error'.

        tracked_file_key : str, optional
            The key of the file to be tracked for checkpoints.
            The default value is 'mosaic'.

        checkpoint_subdir : str, optional
            The name of the subdirectory where checkpoints will be saved.
            The default value is 'checkpoints'.

        checkpoint_selection : list[str], optional
            A list of file keys to be included in the checkpoints.
            The default value is ['mosaic', 'settings', 'log'].

        checkpoint_tag : str, optional
            The tag format to be used for checkpoint file names.
            The default value is '_i{:06d}'.

        checkpoint_freq : int, optional
            The frequency at which checkpoints will be saved.
            The default value is 100.
        '''

        super().__init__(
            input_dir=input_dir,
            input_description=input_description,
            output_dir=output_dir,
            output_description=output_description,
            root_dir=root_dir,
            file_exists=file_exists,
            checkpoint_file_key=tracked_file_key,
            checkpoint_subdir=checkpoint_subdir,
            checkpoint_selection=checkpoint_selection,
            checkpoint_tag=checkpoint_tag,
            checkpoint_freq=checkpoint_freq,
        )

    def open_dataset(self):
        '''Load the mosaic dataset.

        It's kind of awkard that this is one of the only convenience functions
        for opening/loading data.
        '''

        return GDALDatasetIO.load(
            self.output_filepaths['mosaic'],
            mode=gdal.GA_Update,
        )

    def save_to_checkpoint(
        self,
        i: int,
        dataset: gdal.Dataset,
        y_pred: pd.DataFrame = None
    ):
        '''Saves the dataset to a checkpoint file and performs additional
        operations if necessary.

        Parameters
        ----------
        i : int
            The current iteration number.
        dataset : gdal.Dataset
            The GDAL dataset object to be saved.
        y_pred : pd.DataFrame, optional
            The predicted values to be saved as a CSV file. Default is None.
        '''

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

    def load_from_checkpoint(self, i_checkpoint: int) -> dict:
        '''Load data from a checkpoint file.

        Parameters
        ----------
        i_checkpoint : int
            The index of the checkpoint file to load.

        Returns
        -------
        loaded_data : dict
            A dictionary containing the loaded data.
            The dictionary has the following key:
            - 'y_pred': A pandas DataFrame containing the predictions.
        '''

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


class SequentialMosaicIOManager(MosaicIOManager):
    '''Class for managing input and output files for sequential mosaics.
    '''

    def __init__(
        self,
        output_description: dict = {
            'mosaic': 'mosaic.tiff',
            'settings': 'settings.yaml',
            'log': 'log.csv',
            'y_pred': 'y_pred.csv',
            'progress_images_dir': 'progress_images',
            'referenced_images': 'referenced_images/img_ind{:06d}.tiff',
        },
        checkpoint_selection: list[str] = [
            'mosaic', 'settings', 'log', 'y_pred'],
        *args, **kwargs
    ):
        '''Initialize the IOManager object.

        Parameters
        ----------
        output_description : dict, optional
            A dictionary specifying the output file names and their
            default values. The keys represent the file types, and the values
            represent the default file names.
            Default is:
            {
                'mosaic': 'mosaic.tiff',
                'settings': 'settings.yaml',
                'log': 'log.csv',
                'y_pred': 'y_pred.csv',
                'progress_images_dir': 'progress_images',
                'referenced_images': 'referenced_images/img_ind{:06d}.tiff',
            }
        checkpoint_selection : list[str], optional
            A list of file types to be included in the checkpoint.
            Default is ['mosaic', 'settings', 'log', 'y_pred'].
        *args, **kwargs
            Additional arguments and keyword arguments.
        '''
        super().__init__(
            output_description=output_description,
            checkpoint_selection=checkpoint_selection,
            *args, **kwargs
        )


class TrainMosaicIOManager(MosaicIOManager):
    '''Class for managing input and output files for training mosaics,
    i.e. mosaics used as the basis for a sequential mosaic.'''

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
        '''Initialize the IOManager object.

        Parameters
        ----------
        output_description : dict, optional
            A dictionary specifying the output file names for different
            outputs. The default is
            {
                'mosaic': 'mosaic.tiff',
                'settings': 'settings_train.yaml',
                'log': 'log_train.yaml',
                'y_pred': 'y_pred_train.csv',
                'progress_images_dir_train': 'progress_images_train',
            }.
        file_exists : str, optional
            A string indicating the file existence status.
            The default is 'pass'.
        *args, **kwargs
            Additional arguments and keyword arguments.
        '''
        super().__init__(
            output_description=output_description,
            file_exists=file_exists,
            *args, **kwargs
        )


class ReferencedRawSplitter:
    '''Class used for splitting referenced (test + training) and
    raw (production) data.'''

    def __init__(
        self,
        io_manager,
        test_size: Union[int, float] = 0.2,
        max_raw_size: int = None,
        drop_raw_images: bool = False,
        random_state: Union[int, np.random.RandomState] = None,
        use_test_dir: bool = False,
    ):
        '''Initialize the ReferencedRawSplitter object.

        Parameters
        ----------
        io_manager : object
            The IOManager object used for managing input and output files.

        test_size : int or float, optional
            The proportion of the dataset to include in the test split.
            Default is 0.2.

        max_raw_size : int, optional
            The maximum number of raw images to include in the dataset.
            Default is None.

        drop_raw_images : bool, optional
            Whether to drop raw images from the dataset.
            Default is False.

        random_state : int or np.random.RandomState, optional
            The random state used for shuffling the dataset.
            Default is None.

        use_test_dir : bool, optional
            If True, the test data will be determined by a test directory,
            rather than a random split.
            Default is False.
        '''

        self.io_manager = io_manager
        self.test_size = test_size
        self.max_raw_size = max_raw_size
        self.drop_raw_images = drop_raw_images
        self.random_state = check_random_state(random_state)
        self.use_test_dir = use_test_dir

    def train_test_production_split(
        self
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        '''Split the dataset into training, test, and production data.

        How indices are handled:
        - fps_train has indices running from 0 to len(fps_train)
        - fps has indices running from 0 to len(fps)
        - fps_test has indices that are a subset of fps
        '''

        referenced_fps = self.io_manager.input_filepaths['referenced_images']

        # Actual train test split
        if self.use_test_dir:
            fps_test = \
                self.io_manager.input_filepaths['test_referenced_images']
            fps_train = referenced_fps
        else:
            fps_train, fps_test = train_test_split(
                referenced_fps,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
            )

        # Combine raw fps and test fps
        if not self.drop_raw_images:
            raw_fps = self.io_manager.input_filepaths['images']

            # Downsample the raw images as requested
            if self.max_raw_size is not None:
                if raw_fps.size > self.max_raw_size:
                    raw_fps = pd.Series(self.random_state.choice(
                        raw_fps, self.max_raw_size))

            raw_fps.index += referenced_fps.size
            fps = pd.concat([fps_test, raw_fps])
        else:
            fps = fps_test

        fps = fps.sample(
            n=fps.size,
            replace=False,
            random_state=self.random_state,
        )

        return fps_train, fps_test, fps
