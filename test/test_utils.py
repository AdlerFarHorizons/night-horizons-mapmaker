'''Test file for preprocessing.
'''

import os
import unittest

import numpy as np
import pandas as pd

import night_horizons.utils as utils


class TestLoadImage(unittest.TestCase):

    def test_tiff(self):

        fp = '/data/test_data/referenced_images/Geo 843083290_1.tif'
        img = utils.load_image(fp)

        assert len(img.shape) == 3

    def test_raw(self):

        fp = ('/data/test_data/images/23085686/'
              '20220413_221313_1020286912_0_50_3.raw')
        img = utils.load_image(fp)

        assert len(img.shape) == 3

        # Max should be close to 255
        assert img.max() > 250


class TestDiscoverData(unittest.TestCase):

    def setUp(self):

        self.expected_fps_raw = [
            ('/data/test_data/images/23085686/'
             '20220413_221313_1020286912_0_50_3.raw'),
            ('/data/test_data/images/23085687/'
             '20220413_202740_745696_1_50_0.raw'),
            ('/data/test_data/images/23085687/'
             'Geo 836109848_1.tif'),
        ]

    def test_discover_data(self):

        image_dir = '/data/test_data/images'

        fps = utils.discover_data(image_dir)
        assert list(fps) == self.expected_fps_raw

    def test_discover_data_exts(self):

        image_dir = '/data/test_data'

        fps = utils.discover_data(image_dir, extension=['raw', 'tif', 'tiff'])

        actual_fps_a = [_ for _ in list(fps) if 'test_data/images' in _]
        assert list(actual_fps_a) == self.expected_fps_raw

        actual_fps_b = [_ for _ in list(fps) if 'referenced_images' in _]
        assert len(actual_fps_b) > 0

    def test_discover_data_pattern(self):

        image_dir = '/data/test_data/referenced_images'
        expected_fps = [
            '/data/test_data/referenced_images/Geo 843083290_1.tif',
            '/data/test_data/referenced_images/Geo 836109848_1.tif',
        ]

        fps = utils.discover_data(
            image_dir,
            extension=['.tif', '.tiff'],
            pattern=r'Geo\s\d+_\d.tif'
        )
        assert list(fps) == expected_fps


class TestStoreParameters(unittest.TestCase):

    def test_functional(self):

        class MyClass:

            @utils.store_parameters
            def __init__(self, a, b, c=5, dog=True):
                pass

        a = 'sentence'
        b = 'another'
        c = 3
        my_class = MyClass(a, b, c) 

        assert my_class.a == 'sentence'
        assert my_class.b == b
        assert my_class.c == 3


class TestUpdateRow(unittest.TestCase):

    def test_functional(self):

        # Make test dataframe
        rng = np.random.default_rng()
        df = pd.DataFrame(
            rng.uniform(size=(10, 3)),
        )
        df.index = pd.date_range('2022-01-01', periods=10, freq='D')
        df['class'] = 'a'
        original_columns = df.columns

        # Test row
        new_row = df.iloc[3]
        new_row[0] = -1.0
        new_row['new_class'] = 'c'
        new_row['score'] = 0.5

        expected_columns = pd.Index(pd.concat([
            original_columns.to_series(),
            pd.Series(['new_class', 'score'])
        ]))

        # Function call
        df = utils.update_row(df, new_row)

        # Check
        pd.testing.assert_index_equal(df.columns, expected_columns)
        pd.testing.assert_series_equal(
            new_row,
            df.loc[new_row.name]
        )
