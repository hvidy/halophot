from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds
from setuptools import setup

long_description = ''

setup(name='halophot',
      version='0.6.2.3',
      description='K2 halo photometry with total variation.',
      long_description='A Python package for doing photometry of very bright stars in the Kepler/K2 mission using halo photometry, constructing the light curve as a linear combination of pixels.\
      We minimize total variation (TV) of the final light curve with respect to the weights of the individual pixels using analytic gradient descent.\
      The full method is documented in our paper at http://arxiv.org/abs/1708.07462.',
      author='Tim White and Benjamin Pope',
      author_email='benjamin.pope@nyu.edu',
      url='https://github.com/hvidy/halophot',
      package_dir={'halophot':'src'},
      scripts=['bin/halo'],
      packages=['halophot'],
      install_requires=["numpy","matplotlib","astropy", "scipy","autograd","future==0.16.0"],
      license='GPLv3',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Programming Language :: Python"
      ]
     )
