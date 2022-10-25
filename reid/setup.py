# coding:utf-8

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

setup(
    ext_modules=cythonize(Extension(
        'necython',
        sources=[
            'reid/necython/extension.pyx',
            'reid/necython/cpp/aco.cpp',
            'reid/necython/cpp/common.cpp',
            'reid/necython/cpp/sampling.cpp',
            'reid/necython/cpp/walker.cpp',
        ],
        extra_compile_args = ['-std=c++11'],
        language='c++',
    )),
    include_dirs = [numpy.get_include()]
)