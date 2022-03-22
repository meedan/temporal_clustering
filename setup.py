from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='temporal_clustering',
    ext_modules = cythonize('temporal_clustering/_temporal_clustering.pyx',language="c++"),
    include_dirs=[numpy.get_include()],
    packages=["temporal_clustering"]
)

