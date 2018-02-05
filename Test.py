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

start = time.time( )

arcv = mfi_arcv_hres( )

( mfi_t, mfi_b_x, mfi_b_y,
  mfi_b_z ) = arcv.load_rang('2008-03-01', 100 )


# Establish the number of data.

n_mfi = len( mfi_t )

# Compute and store derived paramters.

mfi_s = [ ( t - mfi_t[0] ).total_seconds( )
                                       for t in mfi_t ]

# Compute the vector magnetic field.

mfi_b_vec = [ [ mfi_b_x[i], mfi_b_y[i], mfi_b_z[i]  ]
                for i in range( len( mfi_s ) )      ]

# Compute the magnetic field magnitude.

mfi_b = [ sqrt( mfi_b_x[i]**2 +
                     mfi_b_y[i]**2 +
                     mfi_b_z[i]**2 )
                     for i in range( len( mfi_b_x ) ) ]

# Compute the average magetic field and its norm.

mfi_avg_vec = array( [ mean( mfi_b_x ),
                       mean( mfi_b_y ),
                       mean( mfi_b_z ) ] )

mfi_avg_mag = sqrt( mfi_avg_vec[0]**2 +
                    mfi_avg_vec[1]**2 +
                    mfi_avg_vec[2]**2   )

mfi_avg_nrm = mfi_avg_vec / mfi_avg_mag

mfi_nrm     = [ ( mfi_b_x[i], mfi_b_y[i],
                  mfi_b_z[i] ) /mfi_b[i]
                  for i in range( len( mfi_b ) ) ]

#  Define the sinusoidal model for the data.

def model( x, b0, db, w, p ) :

	b = [ 0. for d in range( len( mfi_s ) ) ]

	b = [ b0 + db*cos( w*mfi_s[i] + p )
	                   for i in range( len( mfi_s ) ) ]
	return b

# Calculate average magnetic field.

avb  = [ sum( [ mfi_b_vec[i][j]
         for i in range( len( mfi_s ) ) ] )/
         ( len(mfi_s ) ) for j in range ( 3 ) ]

# Compute the standard deviation of magnetic field.

davb = [ std( array( [ mfi_b_vec[i][j]
         for i in range( len( mfi_b_vec ) ) ] ) )
         for j in range( 3 )                         ]

# Compute the modeled magnetic field.

mdl_b_vec = transpose( [ model( [ mfi_b_vec[i][j]
                           for i in range( len( mfi_s ) ) ],
                           avb[j], davb[j], 0.1, 50. )
                           for j in range( 3 ) ]               )

mdl_b_vec = [ [ mdl_b_vec[j][i] for i in range( 3 ) ]
                for j in range( len( mfi_s )  ) ]

mdl_b_x = [ mdl_b_vec[i][0] for i in range( len( mfi_s ) ) ]
mdl_b_y = [ mdl_b_vec[i][1] for i in range( len( mfi_s ) ) ]
mdl_b_z = [ mdl_b_vec[i][2] for i in range( len( mfi_s ) ) ]


#def models(x, b0,db,w,p):
#
##	bv = [ [ 0. for i in range( 3 ) ] for d in range( len( mfi_s ) ) ]
#
#	bv = [ [ b0[j] + db[j]*cos( w*mfi_s[i] + p )
#	       for j in range( 3 ) ]
#	       for i in range( len( mfi_s ) ) ]
#
#	return bv
#
#mdl_b_vec = models( mfi_b_vec, avb, davb, 0., 0.)
#
#( fit, covar ) = curve_fit(
#                         models, mfi_b_vec, mdl_b_vec, maxfev = 5000 )
#print fit

( fitx, covarx ) = curve_fit(
                         model, mfi_b_x, mdl_b_x, maxfev=2000 )
( fity, covary ) = curve_fit(
                         model, mfi_b_y, mdl_b_y, maxfev=2000 )
( fitz, covarz ) = curve_fit(
                         model, mfi_b_z, mdl_b_z, maxfev=2000 )

print fitx

mfi_b_x_m = [ fitx[0] + fitx[1]*cos(fitx[2]*mfi_s[i] +
                   fitx[3] ) for i in range( len(mfi_s )) ]
mfi_b_y_m = [ fity[0] + fity[1]*cos(fity[2]*mfi_s[i] +
                   fity[3] ) for i in range( len(mfi_s )) ]
mfi_b_z_m = [ fitz[0] + fitz[1]*cos(fitz[2]*mfi_s[i] +
                   fitz[3] ) for i in range( len(mfi_s )) ]

avb_x  = fitx[0]
avb_y  = fity[0]
avb_z  = fitz[0]

avb_vec = [ fitx[0], fity[0], fitz[0] ]

davb_x = fitx[1]
davb_y = fity[1]
davb_z = fitz[1]

davb_vec = [ fitx[1], fity[1], fitz[1] ]

mfi_omega = sum( fitx[0]*fitx[2] + fity[0]*fity[2] +
                      fitz[0]*fitz[2] ) / sum( avb_vec )

mfi_phi   = 180*( sum( fitx[0]*fitx[3] + fity[0]*fity[3] +
                 fitz[0]*fitz[3] )/ ( pi*sum( avb_vec ) ) )

print 'Computation time = ','%.6f'% (time.time()-start), 'seconds.'
