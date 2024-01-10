# setup.py

from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Build import cythonize
import numpy as np
import sys
import platform


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()


EXTRA_COMPILE_ARGS = ['-std=c++11', '-O2']

if platform.machine() == 'x86_64':
    EXTRA_COMPILE_ARGS += ['-mavx', '-mavx2', '-mfma']

ext = cythonize([
    Extension('cython_trend_lines.trend_lines',
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=[],
        sources=['cython_trend_lines/trend_lines.pyx',
            'cython_trend_lines/covariance.cpp',
            ],
        include_dirs=['cython_trend_lines/', np.get_include()],
        language='c++'),
    ])

setup(
    name="cython_trend_lines",
    author="Mohammad Ghoddosi",
    install_requires=requirements,
    python_requires=">=3.9.5, <4",
    packages=find_packages(),
    ext_modules=ext
)