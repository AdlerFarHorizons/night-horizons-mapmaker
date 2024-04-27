'''Module for exceptions in the night_horizons package.
'''

class HomographyTransformError(ValueError):
    '''An error in the homography transformatin process, typically
    when the homography matrix's determinant has a very large or very
    small value.'''
    pass


class SrcDarkFrameError(ValueError):
    '''An error that occurs when the source image is too dark.
    '''
    pass


class DstDarkFrameError(ValueError):
    '''An error that occurs when the destination image is too dark.
    '''
    pass


class OutOfBoundsError(ValueError):
    '''An error that occurs when a calculation is attempted that is out
    of bounds (typically of the mosaic).
    '''
    pass
