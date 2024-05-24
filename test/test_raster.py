import os
import unittest
import shutil

import numpy as np
import cv2

from night_horizons import raster


class TestImage(unittest.TestCase):

    def setUp(self):

        self.rng = np.random.default_rng(10326)
        self.temp_fp = '/data/output/other/temp.tiff'

    def tearDown(self):
        if os.path.isfile(self.temp_fp):
            os.remove(self.temp_fp)

    def test_open(self):

        example_fp = '/data/input/nitelite.referenced-images/220513-FH135/Geo 836109848_1.tif'
        image = raster.Image.open(example_fp)

        assert image.img is not None

    def test_save_and_open(self):

        # Create a test image
        img = self.rng.uniform(
            low=0, high=255, size=(100, 80, 3)).astype(np.uint8)
        image = raster.Image(img)

        # Save and open
        image.save(self.temp_fp)
        new_image = raster.Image.open(self.temp_fp)

        np.testing.assert_allclose(
            image.img_int,
            new_image.img_int,
        )
        np.testing.assert_allclose(
            image.img,
            new_image.img,
        )

    def test_consistent_input_given_float(self):

        img = self.rng.uniform(low=0, high=1., size=(100, 80, 3))

        image = raster.Image(img)
        np.testing.assert_allclose(
            img,
            image.img,
        )

    def test_consistent_input_given_int(self):

        img_int = self.rng.uniform(
            low=0,
            high=255,
            size=(100, 80, 3),
        ).astype(np.uint8)

        image = raster.Image(img_int)
        np.testing.assert_allclose(
            img_int,
            image.img_int,
        )


class TestReferencedImage(unittest.TestCase):

    def setUp(self):

        x_bounds = np.array([-9599524.7998918, -9590579.50992268])
        y_bounds = np.array([4856260.998546081, 4862299.303607852])

        self.rng = np.random.default_rng(10326)

        img = self.rng.uniform(
            low=0,
            high=255,
            size=(100, 80, 3),
        ).astype(np.uint8)

        self.reffed = raster.ReferencedImage(
            img=img,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        self.temp_fp = '/data/output/other/temp.tiff'
        os.makedirs(os.path.dirname(self.temp_fp), exist_ok=True)
        if os.path.isfile(self.temp_fp):
            os.remove(self.temp_fp)

    def tearDown(self):

        if os.path.isfile(self.temp_fp):
            os.remove(self.temp_fp)

    def test_save_and_open(self):

        self.reffed.save(self.temp_fp)
        new_reffed = raster.ReferencedImage.open(self.temp_fp)

        np.testing.assert_allclose(
            self.reffed.img_int,
            new_reffed.img_int,
        )

    def test_get_latlon_bounds(self):

        lon_bounds, lat_bounds = self.reffed.latlon_bounds

        assert lon_bounds[1] > lon_bounds[0]
        assert lat_bounds[1] > lat_bounds[0]

    def test_get_cart_bounds(self):

        x_bounds, y_bounds = self.reffed.cart_bounds

        assert x_bounds[1] > x_bounds[0]
        assert y_bounds[1] > y_bounds[0]

    def test_show_in_cart_crs(self):

        self.reffed.show(crs='cartesian')
        self.reffed.show(crs='pixel')

    def test_convert_pixel_to_cart(self):

        xs, ys = self.reffed.get_cart_coordinates()
        pxs, pys = self.reffed.get_pixel_coordinates()

        actual_xs, actual_ys = self.reffed.convert_pixel_to_cart(
            pxs,
            pys
        )

        np.testing.assert_allclose(xs, actual_xs)
        np.testing.assert_allclose(ys, actual_ys)

    def test_convert_cart_to_pixel(self):

        xs, ys = self.reffed.get_cart_coordinates()
        pxs, pys = self.reffed.get_pixel_coordinates()

        actual_pxs, actual_pys = self.reffed.convert_cart_to_pixel(xs, ys)

        np.testing.assert_allclose(pxs, actual_pxs)
        np.testing.assert_allclose(pys, actual_pys)
