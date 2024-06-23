import setuptools
import numpy as np
from Cython.Build import cythonize

install_deps = ["importlib-metadata",
        "natsort",
        "rastermap>=0.9.0",
        "tifffile",
        "torch>=1.13.1",
        "numpy==1.26",
        "numba>=0.57.0",
        "matplotlib",
        "scipy>=1.9.0",
        "scikit-learn",
        "cellpose",
        "scanimage-tiff-reader>=1.4.1",
        "h5py",
        "qtpy",
        "pyqt6",
        "pyqt6.sip",
        "pyqtgraph",
        "jupyter",
        "ipympl",
        "scikit-image",
        ]


external_modules = [setuptools.extension.Extension("suite2p.extraction.oasis",
                                                   sources=["suite2p/extraction/oasis.pyx"],
                                                   include_dirs=[np.get_include()],
                                                   language="c++"
                                                   )]

try:
    import torch
    a = torch.ones(2, 3)
    major_version, minor_version, _ = torch.__version__.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except:
    pass

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="suite2p",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="marius10p@gmail.com",
    description="Pipeline for calcium imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/suite2p",
    packages=setuptools.find_packages(),
    setup_requires=[
      "numpy==1.26",
    ],
    install_requires=install_deps,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        "console_scripts": [
            "suite2p = suite2p.__main__:main",
        ]
    },
    ext_modules=cythonize(external_modules, language_level="3"),
)
