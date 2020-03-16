import halophot
from halophot.halo_tools import halo_tpf
import lightkurve as lk

import os
TESTDIR = os.path.abspath(os.path.dirname(__file__))
ddir = os.path.join(TESTDIR,'../data/')

def test_lk():

	fname = ddir+"ktwo205897543-c03_lpd-targ.fits.gz"

	tpf = halo_tpf(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	meta, corr_lc = tpf.halo(thresh=0.5)

def test_k2sc():
	import k2sc
	from k2sc import standalone
	print('k2sc version',k2sc.__version__)

	fname = ddir+"ktwo205897543-c03_lpd-targ.fits.gz"

	tpf = lk.KeplerTargetPixelFile(fname)

	lc = tpf.to_lightcurve()
	lc.primary_header = tpf.hdu[0].header
	lc.data_header = tpf.hdu[1].header

	lc.__class__ = standalone.k2sc_lc

	lc.k2sc(de_max_time=10)

# def test_saturation():
# 	print('testing')

# 	fname = ddir+"ktwo200128910-c111_lpd-targ.fits.gz" 

# 	tpf = halo_tpf(fname)

# 	lc = tpf.to_lightcurve()
# 	lc.primary_header = tpf.hdu[0].header
# 	lc.data_header = tpf.hdu[1].header

# 	meta, corr_lc = tpf.halo(thresh=-1);
# 	weightmap = meta['weightmap'][0]
