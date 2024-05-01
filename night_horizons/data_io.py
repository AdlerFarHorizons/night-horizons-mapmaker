"""This module provides classes for data input/output operations.
"""

from abc import ABC, abstractmethod
import os
from typing import Tuple, Union

import cv2
import numpy as np
import osgeo
from osgeo import gdal, gdal_array
from osgeo.osr import SpatialReference
import pandas as pd
import pyproj
from pyproj.enums import WktVersion

gdal.UseExceptions()


class DataIO(ABC):
    """Abstract base class for all data input/output"""

    @staticmethod
    def save(filepath: str, data: object):
        """Save data to disk.

        Parameters
        ----------
            filepath : str
                Location to save the data.

            data :
                The data to save.
        """

    @staticmethod
    def load(filepath: str) -> object:
        """Load data from a file.

        Parameters
        ----------
        filepath : str
            The path to the file to be loaded.

        Returns
        -------
        data : object
            The loaded data.
        """


class ImageIO(DataIO):
    """Class for loading and saving images."""

    name = "image"

    @staticmethod
    def save(filepath: str, data: np.ndarray):
        """Save an image to disk.

        Parameters
        ----------
            filepath : str
                Location to save the data.

            data : np.ndarray
                The image to save.
        """

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = data[:, :, ::-1]
        cv2.imwrite(filepath, data)

    @staticmethod
    def load(
        filepath: str,
        dtype: Union[str, type] = "uint8",
        img_shape: Tuple[int] = (1200, 1920),
    ) -> np.ndarray:
        """Load an image from disk.

        Parameters
        ----------
            filepath : str
                Location of the image.
            dtype : Union[str, type], optional
                Datatype. Defaults to integer from 0 to 255.
            img_shape : Tuple[int], optional
                Image dimensions, used if loading a file of the '.raw' type.

        Returns
        -------
            img : np.ndarray
                Image as a numpy array.
        """

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        ext = os.path.splitext(filepath)[1]

        # Load and reshape raw image data.
        if ext == ".raw":

            raw_img = np.fromfile(filepath, dtype=np.uint16)
            raw_img = raw_img.reshape(img_shape)

            img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)
            img_max = 2**12 - 1

        elif ext in [".tiff", ".tif"]:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            # CV2 defaults to BGR, but RGB is more standard for our purposes
            img = img[:, :, ::-1]
            img_max = np.iinfo(img.dtype).max

        else:
            raise IOError("Cannot read filetype {}".format(ext))

        if isinstance(dtype, str):
            dtype = getattr(np, dtype)

        # Check if conversion needs to be done
        if img.dtype == dtype:
            return img

        # Rescale
        img = img / img_max
        img = (img * np.iinfo(dtype).max).astype(dtype)

        return img


class GDALDatasetIO(DataIO):
    """Class for reading and writing raster datasets using GDAL.

    Attributes
    ----------
    name : str
        Used for distinguishing between different data input/output classes.
    """

    name = "gdal_dataset"

    @staticmethod
    def save(filepath: str, data: gdal.Dataset, driver: str = "GTiff"):
        """Save a raster dataset to a file.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        data : gdal.Dataset
            The raster dataset to be saved.
        driver : str, optional
            The GDAL driver to be used for saving the dataset.
            Defaults to 'GTiff'.
        """

        # Create a copy with a driver that saves to disk
        driver = gdal.GetDriverByName("GTiff")
        save_dataset = driver.CreateCopy(filepath, data, 0)
        save_dataset.FlushCache()
        save_dataset = None

    @staticmethod
    def load(
        filepath: str, mode: int = gdal.GA_ReadOnly, crs: pyproj.CRS = None
    ) -> gdal.Dataset:
        """Load a raster dataset from a file.

        Parameters
        ----------
        filepath : str
            Path to the input file.
        mode : int, optional
            GDAL access mode. Defaults to gdal.GA_ReadOnly.
        crs : pyproj.CRS, optional
            Desired coordinate system. Defaults to None.

        Returns
        -------
        gdal.Dataset
            The loaded raster dataset.
        """

        data = gdal.Open(filepath, mode)

        # Convert to desired crs.
        if crs is not None:
            data = GDALDatasetIO.convert(data, crs)

        return data

    @staticmethod
    def load_from_viirs_hdf5(
        filepath: str,
        output_filepath: str = None,
    ) -> gdal.Dataset:
        """Load a VIIRS HDF5 file and convert it to a raster dataset.

        Parameters
        ----------
        filepath : str
            Path to the VIIRS HDF5 file.
        output_filepath : str, optional
            Path to the output raster file. If not provided, a default
            output file path will be used.

        Returns
        -------
        gdal.Dataset
            The loaded and converted raster dataset.
        """

        if output_filepath is None:
            output_filepath = filepath.replace(".h5", ".tiff")

        # If the conversion is already done
        if os.path.isfile(output_filepath):
            return GDALDatasetIO.load(output_filepath)

        # Open HDF file
        hdflayer = gdal.Open(filepath, gdal.GA_ReadOnly)

        # Open raster layer
        # hdflayer.GetSubDatasets()[0][0] - for first layer
        # hdflayer.GetSubDatasets()[1][0] - for second layer ...etc
        subhdflayer = hdflayer.GetSubDatasets()[0][0]
        rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

        # collect bounding box coordinates
        horizontal_tile_number = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
        vertical_tile_number = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])

        west_bound_coord = (10 * horizontal_tile_number) - 180
        north_bound_coord = 90 - (10 * vertical_tile_number)
        east_bound_coord = west_bound_coord + 10
        south_bound_coord = north_bound_coord - 10

        # WGS84
        epsg = "-a_srs EPSG:4326"

        translate_option_text = (
            epsg
            + " -a_ullr "
            + str(west_bound_coord)
            + " "
            + str(north_bound_coord)
            + " "
            + str(east_bound_coord)
            + " "
            + str(south_bound_coord)
        )
        translateoptions = gdal.TranslateOptions(
            gdal.ParseCommandLine(translate_option_text)
        )

        translated = gdal.Translate(output_filepath, rlayer, options=translateoptions)

        return translated

    @staticmethod
    def create(
        filepath: str,
        x_min: float,
        y_max: float,
        pixel_width: float,
        pixel_height: float,
        crs: pyproj.CRS,
        x_size: int,
        y_size: int,
        n_bands: int = 4,
        driver: str = "MEM",
        return_dataset: bool = True,
        options: list[str] = None,
        *args,
        **kwargs,
    ):
        """
        Create an empty raster dataset.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        x_min : float
            Minimum x-coordinate of the dataset.
        y_max : float
            Maximum y-coordinate of the dataset.
        pixel_width : float
            Width of each pixel in the dataset.
        pixel_height : float
            Height of each pixel in the dataset.
        crs : pyproj.CRS
            Coordinate reference system (CRS) of the dataset.
        x_size : int
            Number of pixels in the x-direction.
        y_size : int
            Number of pixels in the y-direction.
        n_bands : int, optional
            Number of bands in the dataset. Defaults to 4.
        driver : str, optional
            The GDAL driver to be used for creating the dataset.
            Defaults to 'MEM'.
        return_dataset : bool, optional
            Whether to return the created dataset. Defaults to True.
        options : list[str], optional
            Additional options for creating the dataset.
            Defaults to ['TILED=YES'].
        *args, **kwargs
            Additional arguments and keyword arguments to be passed to
            the GDAL driver.

        Returns
        -------
        gdal.Dataset or None
            The created raster dataset if `return_dataset` is True,
            otherwise None.
        """

        if options is None:
            options = ["TILED=YES"]

        # Initialize an empty GeoTiff
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(
            filepath,
            xsize=x_size,
            ysize=y_size,
            bands=n_bands,
            options=options,
            *args,
            **kwargs,
        )

        # Properties
        dataset.SetProjection(crs.to_wkt())
        dataset.SetGeoTransform(
            [
                x_min,
                pixel_width,
                0.0,
                y_max,
                0.0,
                pixel_height,
            ]
        )
        if n_bands == 4:
            dataset.GetRasterBand(4).SetMetadataItem("Alpha", "1")

        if (driver == "MEM") or return_dataset:
            return dataset

        # Close out the dataset for now. (Reduces likelihood of mem leaks.)
        dataset.FlushCache()
        dataset = None

    @staticmethod
    def convert(dataset: gdal.Dataset, target_crs: pyproj.CRS) -> gdal.Dataset:
        """Convert the coordinate reference system (CRS) of a raster dataset.

        Parameters
        ----------
        dataset : gdal.Dataset
            The raster dataset to be converted.
        target_crs : pyproj.CRS
            The target coordinate reference system (CRS) for the conversion.

        Returns
        -------
        gdal.Dataset
            The converted raster dataset.
        """

        # Check current CRS
        current_crs = pyproj.CRS(dataset.GetProjection())
        if current_crs == target_crs:
            return dataset

        # Change CRS to SRS so gdal knows what to do
        srs = SpatialReference()
        if osgeo.version_info.major < 3:
            srs.ImportFromWkt(target_crs.to_wkt(WktVersion.WKT1_GDAL))
        else:
            srs.ImportFromWkt(target_crs.to_wkt())

        converted_dataset = gdal.Warp(
            "",
            dataset,
            format="MEM",
            dstSRS=srs,
        )

        return converted_dataset

    @staticmethod
    def get_bounds_from_dataset(
        dataset: gdal.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, pyproj.CRS]:
        """Get the image bounds in a given coordinate system.

        Parameters
        ----------
        dataset : gdal.Dataset
            The raster dataset.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, float, pyproj.CRS]
            The x and y bounds of the image in the target coordinate system,
            pixel width, pixel height, and the coord reference system (CRS).
        """

        # Get the coordinates
        x_min, pixel_width, x_rot, y_max, y_rot, pixel_height = (
            dataset.GetGeoTransform()
        )

        # Get bounds
        x_max = x_min + pixel_width * dataset.RasterXSize
        y_min = y_max + pixel_height * dataset.RasterYSize
        x_bounds = [x_min, x_max]
        y_bounds = [y_min, y_max]

        return (
            x_bounds,
            y_bounds,
            pixel_width,
            pixel_height,
        )


class ReferencedImageIO(DataIO):
    name = "referenced_image"

    @staticmethod
    def save(
        filepath: str,
        img: np.ndarray,
        x_bounds: np.ndarray,
        y_bounds: np.ndarray,
        crs: pyproj.CRS,
        driver: str = "GTiff",
        *args,
        **kwargs,
    ):
        """
        Save an image to a file.

        Parameters
        ----------
        filepath : str
            The path to the output file.
        img : np.ndarray
            The image data to be saved.
        x_bounds : np.ndarray
            The x-coordinates bounds of the image.
        y_bounds : np.ndarray
            The y-coordinates bounds of the image.
        crs : pyproj.CRS
            The coordinate reference system of the image.
        driver : str, optional
            The GDAL driver to use for saving the image. Defaults to 'GTiff'.
        *args, **kwargs
            Additional arguments to be passed to GDALDatasetIO.create

        Example
        -------
        save('output.tif', image_data, x_bounds, y_bounds, crs)
        """

        if filepath != "":
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Get data type
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(img.dtype)

        # Get pixel size
        pixel_width = (x_bounds[1] - x_bounds[0]) / img.shape[1]
        pixel_height = -(y_bounds[1] - y_bounds[0]) / img.shape[0]

        dataset = GDALDatasetIO.create(
            filepath=filepath,
            x_min=x_bounds[0],
            y_max=y_bounds[1],
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            crs=crs,
            x_size=img.shape[1],
            y_size=img.shape[0],
            n_bands=img.shape[2],
            driver=driver,
            eType=gdal_dtype,
            *args,
            **kwargs,
        )

        # Write to the dataset
        dataset.WriteArray(img.transpose(2, 0, 1))

        # Stop and return if we want to keep the dataset open
        if driver == "MEM":
            return dataset

        dataset.FlushCache()
        dataset = None

    @staticmethod
    def load(
        filepath: str, crs: pyproj.CRS = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a referenced image from a file.

        Parameters
        ----------
        filepath : str
            The path to the image file.
        crs : pyproj.CRS, optional
            The coordinate reference system of the image. Defaults to None.

        Returns
        -------
        tuple
            A tuple containing the loaded image, x bounds, and y bounds.
        """

        # Get image
        dataset = GDALDatasetIO.load(filepath, crs=crs)
        img = dataset.ReadAsArray()

        # For multiple bands, format accordingly
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)

        # Get bounds
        (x_bounds, y_bounds, dx, dy) = GDALDatasetIO.get_bounds_from_dataset(dataset)

        data = (img, x_bounds, y_bounds)

        return data


class TabularIO(DataIO):
    name = "tabular"

    @staticmethod
    def save(filepath: str, data: pd.DataFrame):
        """
        Save data to a CSV file.

        Parameters
        ----------
        filepath : str
            The path to the CSV file where the data will be saved.
        data : pd.DataFrame
            The data to be saved. It can be a list of dictionaries
            or a dictionary of lists.
        """
        data.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath: str) -> pd.DataFrame:
        """Load data from a CSV file.

        Parameters
        ----------
        filepath : str
            The path to the CSV file.

        Returns
        -------
        pandas.DataFrame
            The loaded data as a pandas DataFrame.
        """
        df = pd.read_csv(filepath)
        return df
