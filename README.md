# halophot
[![Build Status](https://travis-ci.org/OxES/k2sc.svg?branch=master)](https://travis-ci.org/OxES/k2sc)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)

## K2 Halo Photometry

This is code that implements a Total Variation (TV) based regularization for Kepler/K2 photometry of very bright stars. 

We minimize nth order TV - i.e. the sum of the absolute values of the nth differences of a light curve - of a light curve created as the weighted sum of pixels, with weights in (0,1). This appears remarkably effective at removing pointing-based systematics from K2 lightcurves where it is impractical to do photometry otherwise and apply more standard detrending methods. 

We believe this is of practical use for Kepler targets brighter than  Kp ~ 6. 

We invite anyone interested to use and modify this code under a GPL v3 license. 