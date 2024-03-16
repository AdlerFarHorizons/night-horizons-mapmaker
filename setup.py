'''Installation script for Night Horizons mapmaker.
'''

import setuptools

setuptools.setup(
    name="night_horizons",
    packages=setuptools.find_packages(),
    install_requires=['ruamel.yaml'],
)
