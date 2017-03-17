# halophot
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)

## Contributors

Tim White, Benjamin Pope

## K2 Halo Photometry

This is code that implements a Total Variation (TV) based regularization for Kepler/K2 photometry of very bright stars. 

We minimize nth order TV - i.e. the sum of the absolute values of the nth differences of a light curve - of a light curve created as the weighted sum of pixels, with weights in (0,1). This appears remarkably effective at removing pointing-based systematics from K2 lightcurves where it is impractical to do photometry otherwise and apply more standard detrending methods. 

We believe this is of practical use for Kepler targets brighter than  Kp ~ 6. 

## Basic usage

First install with python setup.py install --user (or whichever other approach you prefer), and add the appropriate paths to your PYTHONPATH. Call halophot either as a library of functions (/src/halo_tools.py) or from the command line as, e.g.

halo ktwo200007768-c04_lpd-targ.fits --data-dir /path/to/data/directory/ --name Atlas -c 4 --do-plot -sub 8

where --sub is the subsampling parameter you would like to use (for science, use 1; for quick tests, it can be useful to run faster with subsampling) and -c is the campaign.

## Theano

We achieve a dramatic speed-up using Theano to compute analytic derivatives. This package can be downloaded at http://deeplearning.net/software/theano/install.html.

## License

We invite anyone interested to use and modify this code under a GPL v3 license. 

## Citation

If you use our code, please cite us! The paper "Breaking the K2 bright limit with halo photometry: variability in the seven brightest stars of the Pleiades" (White et al.) has been submitted to MNRAS and KASC and we recommend that you cite this as in prep., and update the citation accordingly when the paper is (hopefully) accepted.