"""Run: python setup.py build_ext --inplace
"""

import numpy
from Cython.Distutils import build_ext

from distutils.core import setup
from distutils.extension import Extension

setup(
    name = "Cliodynamics",
    cmdclass = {"build_ext": build_ext}, 
    ext_modules = [
        Extension(
            "models.frontier_attacks",
            ["models/frontier_attacks.pyx"],
            include_dirs=[numpy.get_include()]
        ),
    ],
)