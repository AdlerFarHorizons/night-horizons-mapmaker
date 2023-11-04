'''Main mapmaker file.
'''

from . import observations


class Mapmaker:

    def __init__(self, images_dir, metadata_fp):

        self.observing_run = observations.ObservingRun()
