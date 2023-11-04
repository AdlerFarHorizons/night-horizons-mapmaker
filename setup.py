'''Installation script for Night Horizons mapmaker.
'''

import setuptools

setuptools.setup(
    name="night_horizons",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pytest',
        'jupyterlab',
        'jupyter_contrib_nbextensions',
    ],
)
