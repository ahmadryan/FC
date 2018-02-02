# Load the modules necessary for signaling the graphical interface.

from PyQt4.QtCore import QObject, SIGNAL, QThread

# Load the modules necessary for file operations.

import os.path

# Load the modules necessary for handling dates and times.

import time
from time import sleep
from datetime import datetime, timedelta
from janus_time import calc_time_epc, calc_time_sec, calc_time_val

# Load the module necessary handling step functions.

from janus_step import step

# Load the dictionary of physical constants.

from janus_const import const

# Load the modules necessary for loading Wind/FC and Wind/MFI data.

from janus_fc_arcv   import fc_arcv
from janus_spin_arcv import spin_arcv
from janus_mfi_arcv_lres  import mfi_arcv_lres
from janus_mfi_arcv_hres import mfi_arcv_hres

# Load the necessary array modules and mathematical functions.

from numpy import amax, amin, append, arccos, arctan2, arange, argsort, array, \
                    average, cos, deg2rad, diag, dot, exp, indices, interp, \
                    mean, pi, polyfit, rad2deg, reshape, sign, sin, sum, sqrt, \
                    std, tile, transpose, where, zeros, shape

from numpy.linalg import lstsq

from scipy.special     import erf
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit
from scipy.stats       import pearsonr, spearmanr

from janus_helper import round_sig

from janus_fc_spec import fc_spec

# Load the "pyon" module.

from janus_pyon import plas, series

# Load the modules necessary for saving results to a data file.

import pickle

# Load the modules necessary for copying.

from copy import deepcopy


# Establish the number of data.

self.n_mfi = len( self.mfi_t )

# Compute and store derived paramters.

self.mfi_s = [ ( t - self.fc_spec['time'] ).total_seconds( )
                                       for t in self.mfi_t ]

# Compute the vector magnetic field.

self.mfi_b_vec = [ [ self.mfi_b_x[i],
                     self.mfi_b_y[i],
                     self.mfi_b_z[i]                     ]
                     for i in range( len( self.mfi_s ) ) ]

# Compute the magnetic field magnitude.

self.mfi_b = [ sqrt( self.mfi_b_x[i]**2 +
                     self.mfi_b_y[i]**2 +
                     self.mfi_b_z[i]**2 )
                     for i in range( len( self.mfi_b_x ) ) ]

# Compute the average magetic field and its norm.

self.mfi_avg_vec = array( [ mean( self.mfi_b_x ),
                            mean( self.mfi_b_y ),
                            mean( self.mfi_b_z ) ] )

self.mfi_avg_mag = sqrt( self.mfi_avg_vec[0]**2 +
                         self.mfi_avg_vec[1]**2 +
                         self.mfi_avg_vec[2]**2   )

self.mfi_avg_nrm = self.mfi_avg_vec / self.mfi_avg_mag

mfi_nrm     = [ ( self.mfi_b_x[i], self.mfi_b_y[i],
                  self.mfi_b_z[i] ) /self.mfi_b[i]
                  for i in range( len( self.mfi_b ) ) ]

# Curve fitting for MFI data.

def model( x, b0, db, w, p ) :

	b = [ 0. for d in range( len( self.mfi_s ) ) ]

	b =  [ b0 + db*cos( w*self.mfi_s[i] + p )
	                   for i in range( len( self.mfi_s ) ) ]

	print b[0]
	return b

avb  = [ sum( [ self.mfi_b_vec[i][j]
         for i in range( len( self.mfi_s ) ) ] )/
         ( len(self.mfi_s ) ) for j in range ( 3 ) ]

davb = [ std( array( [ self.mfi_b_vec[i][j]
         for i in range( len( self.mfi_b_vec ) ) ] ) )
         for j in range( 3 )                         ]

y = transpose( [ model( [ self.mfi_b_vec[i][j]
                           for i in range( len(self.mfi_s ) ) ],
                           avb[j], davb[j], 1., 20. )
                           for j in range( 3 ) ]               )

bx = self.mfi_b_x
test_bx = model( bx, sum(bx)/len(bx), std(array(bx)),0.,0. )
(tfit, tcovar) = curve_fit( model, bx, test_bx, maxfev=5000 )
print tfit, sum(bx)/len(bx), std(array(bx))


y = [ [ y[j][i] for i in range(3)]
                for j in range( len( self.mfi_s)  ) ]

y_x = [ y[i][0] for i in range( len( self.mfi_s ) ) ]
y_y = [ y[i][1] for i in range( len( self.mfi_s ) ) ]
y_z = [ y[i][2] for i in range( len( self.mfi_s ) ) ]

( fitx, covarx ) = curve_fit(
                         model, self.mfi_b_x, y_x, maxfev=5000 )
( fity, covary ) = curve_fit(
                         model, self.mfi_b_y, y_y, maxfev=5000 )
( fitz, covarz ) = curve_fit(
                         model, self.mfi_b_z, y_z, maxfev=5000 )

self.mfi_b_x_m = [ fitx[0] + fitx[1]*cos(fitx[2]*self.mfi_s[i] +
                   fitx[3] ) for i in range( len(self.mfi_s )) ]
self.mfi_b_y_m = [ fity[0] + fity[1]*cos(fity[2]*self.mfi_s[i] +
                   fity[3] ) for i in range( len(self.mfi_s )) ]
self.mfi_b_z_m = [ fitz[0] + fitz[1]*cos(fitz[2]*self.mfi_s[i] +
                   fitz[3] ) for i in range( len(self.mfi_s )) ]

self.avb_x  = fitx[0]
self.avb_y  = fity[0]
self.avb_z  = fitz[0]

self.avb_vec = [ fitx[0], fity[0], fitz[0] ]

self.davb_x = fitx[1]
self.davb_y = fity[1]
self.davb_z = fitz[1]

self.davb_vec = [ fitx[1], fity[1], fitz[1] ]

self.mfi_omega = sum( fitx[0]*fitx[2] + fity[0]*fity[2] +
                      fitz[0]*fitz[2] ) / sum( self.avb_vec )

self.mfi_phi   = 180*( sum( fitx[0]*fitx[3] + fity[0]*fity[3] +
                 fitz[0]*fitz[3] )/ ( pi*sum( self.avb_vec ) ) )
