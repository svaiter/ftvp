from __future__ import absolute_import, division, print_function
import sys
import os
import shutil
import numpy

from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    print("You need to install Cython in order to compile this module.")
    sys.exit()

setup(
    name='pyftvp',
    version='0.1',
    description='Fast Total Variation Proximity operator using CUDA',
    author='Samuel Vaiter',
    author_email='samuel.vaiter@gmail.com',
    url='https://github.com/svaiter/ftvp',
    packages=['pyftvp'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("pyftvp._ftvp",
                  sources=["pyftvp/_ftvp.pyx"],
                  libraries=["ftvp"],          # refers to "libexternlib.so"
                  extra_compile_args=["-I../..", "-fPIC"],
                  extra_link_args=["-L../.."],
                  include_dirs=[numpy.get_include(),'../..','.'],
                  language="c++"
             ),
        Extension("pyftvp._ftvp_color",
                  sources=["pyftvp/_ftvp_color.pyx"],
                  libraries=["ftvp-color"],          # refers to "libexternlib.so"
                  extra_compile_args=["-I../..", "-fPIC"],
                  extra_link_args=["-L../.."],
                  include_dirs=[numpy.get_include(),'../..','.'],
                  language="c++"
             )
        ]
)
