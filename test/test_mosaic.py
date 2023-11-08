'''Test file for mosaic objects.
'''

import os
import unittest

from night_horizons.mosaic import Mosaic

class TestMosaic(unittest.TestCase):

    def test_bounds_to_offset(self):

        mosaic = Mosaic()

        # Bounds for the whole dataset
        (
            x_offset,
            y_offset,
            x_count,
            y_count,
        ) = mos(
            dataset.x_min, dataset.x_max,
            dataset.y_min, dataset.y_max,
        )

        assert x_offset == 0
        assert y_offset == 0
        assert x_count == dataset.RasterXSize
        assert y_count == dataset.RasterYSize

