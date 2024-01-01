import unittest
from night_horizons import io


class TestIOManager(unittest.TestCase):

    def setUp(self):

        self.work_dir = './test/test_data'

        io_manager = io.IOManager(
            work_dir=self.work_dir,
        )

    def test_read_file(self):
        # Add your test code here
        pass

    def test_write_file(self):
        # Add your test code here
        pass
