import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock
from SuzPyUtils.norm import medsig
from psf_sim import *
from k2sc.cdpp import cdpp

import halophot 
from halophot.halo_tools import *
import everest


import matplotlib as mpl

mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=18               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


fname = '../data/ktwo211309989-c05_lpd-targ.fits.gz' # point this path to your favourite K2SC light curve

print 'Running everest'
# try:
# 	pass
# except:

star = everest.Everest(211309989, clobber=True, mission='k2',
                    giter=1, gmaxf=3, lambda_arr=[1e0, 1e5, 1e10], oiter=3,
                    pld_order=2, get_hires=False, get_nearby=False)
	# star.publish()

print 'Running halophot'

tpf, ts = read_tpf(fname)

tpf, newts, weights, weightmap, pixelvector = do_lc(tpf,ts,(None,None),1, 1)