from setuptools import setup, Extension
import numpy
import sys

module = Extension('pyomp',
                   sources=['pyomp.c', 'ompcore.c', 'omputils.c', 'ompprof.c', 'myblas.c'],
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=['-O3'],
                   )

setup(name='pyomp',
      version='1.0',
      description='Python interface for ompcore C function',
      ext_modules=[module])
