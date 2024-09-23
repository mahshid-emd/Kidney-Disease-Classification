import setuptools


__version__ = "0.0.0"
REPO_NAME = "Kidney-Disease-Classification-Deep-Learning-Project"
SRC_REPO = "cnnClassifier"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description="A small python package for CNN app",
    long_description_content="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)