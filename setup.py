import setuptools

__version__ = '0.0'
SRC_REPO = 'cnnClassifier'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description='A small python package for CNN app',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)