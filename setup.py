import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(
    os.path.join(EXTENSION_PATH, "config", "extension.toml")
)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "meshcat",
    "tensordict",
    "scikit-learn",
    "seaborn",
    "python-fcl",
    "pyembree",
    "lightning",
    "gymnasium==0.29.0",
    "urdfpy @ git+https://github.com/meenalparakh/urdfpy.git",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]


# Installation operation
setup(
    name="anybody",
    author="Meenal Parakh",
    maintainer="Meenal Parakh",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["anybody"],
    zip_safe=False,
)
