'''Test file for top-level mapmaking functionality.
'''

import os
import unittest

import numpy as np

import night_horizons_mapmaker.mapmaker as mapmaker


class TestGlobal(unittest.TestCase):

    def test_mapmake(self):

        mm = mapmaker.Mapmaker()

        mm.mapmake()

    def test_load(self):

        mm = mapmaker.Mapmaker()

        mm.load()

    def test_preprocess(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()

    def test_georeference(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()
        mm.georeference()

    def test_construct_mosaic(self):

        mm = mapmaker.Mapmaker()

        mm.load()
        mm.preprocess()
        mm.georeference()
        mm.construct_mosaic()
