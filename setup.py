from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

long_description = ''

setup(name='halophot',
      version='0.5',
      description='K2 halo photometry with total variation.',
      long_description=long_description,
      author='Tim White and Benjamin Pope',
      author_email='',
      url='',
      package_dir={'halophot':'src'},
      scripts=['bin/halo'],
      packages=['halophot'],
      install_requires=["numpy", "astropy", "scipy","autograd"],
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
