import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.optimize as optimize
import fitsio
from time import time as clock
from SuzPyUtils.norm import medsig
from psf_sim import *
from k2sc.cdpp import cdpp

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
colours = mpl.rcParams['axes.color_cycle']


fname = '../EPIC_211309989_mast.fits' # point this path to your favourite K2SC light curve
lc = Table.read(fname)

x, y = lc['x'][150:1550], lc['y'][150:1550] # copy in the xy variations from real data 
xx = np.copy(x)
yy = np.copy(y) # make copies to shuffle

ncad = np.size(x)

'''--------------------------------------------------
halo_permute.py - does halo photometry work 
if you permute all the timestamps? No, it turns out!
--------------------------------------------------'''

'''------------------------
Generate a toy light curve
------------------------'''

t = np.linspace(0,100,ncad)

nsims = 50 

f = 20*np.ones(ncad) + np.sin(t/6.) # make this whatever function you like! 
f[400:500] *= 0.990 # toy transit

ff = np.copy(f)

lcs = np.zeros((ncad,nsims))
sigs_raw = np.zeros(nsims)
sigs_1 = np.zeros(nsims)

start = clock()

for jj in range(nsims):
	# print 'Doing period',period

	args = np.random.permutation(ncad)
	reverse = np.argsort(args)

	xx = x[args]
	yy = y[args]
	ff = f[args]

	raw_lc, lc_opt_1 = do_sim(xx,yy,ff)

	lc_opt_1 = lc_opt_1[reverse]
	raw_lc = raw_lc[reverse]
	lcs[:,jj] = lc_opt_1 

	ssr = cdpp(t,raw_lc-f/np.nanmedian(f)+1)
	ss1 = cdpp(t,lc_opt_1-f/np.nanmedian(f)+1)

	sigs_raw[jj] = ssr
	sigs_1[jj] = ss1


finish = clock()
print 'Done'
print 'Time elapsed:',finish-start,'s'

raw, opt = do_sim(x,y,f)

best = cdpp(t,opt-f/np.nanmedian(f)+1)

plt.figure(0)
plt.clf()

plt.plot(t,opt,'.',label='Real x',color=colours[0])
for j in range(nsims):
	plt.plot(t,lcs[:,j],'.',color=colours[1],alpha=0.2)
plt.plot(t,f/np.nanmedian(f),'-',label='True',color=colours[2])
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(r'True Timestamps vs Permuted')
plt.savefig('permuted_lcs.png')
plt.show()

plt.figure(1)
plt.clf()

h = plt.hist(sigs_1,density=True,label="Permuted",color=colours[1])
plt.axvline(best,label="True",color=colours[0])
plt.xlabel(r'CDPP')
plt.ylabel('Density')
plt.title('CDPPs of Permuted Timeseries')
plt.legend()
plt.savefig('permuted_hist.png')
plt.show()
