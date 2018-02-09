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
                    std, tile, transpose, where, zeros, shape, abs, linspace,\
                    cross, angle

from math import atan, degrees

from numpy.linalg import lstsq, norm
from numpy.fft import rfftfreq

from scipy.special     import erf
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit
from scipy.stats       import pearsonr, spearmanr
from scipy.fftpack     import fft, rfft
from scipy.signal      import medfilt

from janus_helper import round_sig

from janus_fc_spec import fc_spec

# Load the "pyon" module.

from janus_pyon import plas, series

import matplotlib.pyplot as plt

# Load the modules necessary for saving results to a data file.

import pickle

# Load the modules necessary for copying.

from copy import deepcopy

start = time.time( )

arcv = mfi_arcv_hres( )

( mfi_t, mfi_b_x, mfi_b_y,
  mfi_b_z ) = arcv.load_rang('2008-11-04-12-00-00', 100 )

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
z  = [0., 0., 1.]
e1 = mfi_avg_nrm
e2 = cross( z, e1 )/ norm( cross( e1, z ) )
e3 = cross( e1, e2 )

bx = [ sum( [ mfi_b_vec[i][j]*e1[j] for j in range(3)] )
                        for i in range( len( mfi_s ) ) ]
by = [ sum( [ mfi_b_vec[i][j]*e2[j] for j in range(3)] )
                        for i in range( len( mfi_s ) ) ]
bz = [ sum( [ mfi_b_vec[i][j]*e3[j] for j in range(3)] )
                        for i in range( len( mfi_s ) ) ]

b_vec = [ [ bx[i], by[i], bz[i] ] for i in range( len( mfi_s ) ) ]

f_x = rfft( bx )
f_y = rfft( by )
f_z = rfft( bz )

# Compute the standard deviation of magnetic field.

davb = [ std( array( [ mfi_b_vec[i][j]
         for i in range( len( mfi_b_vec ) ) ] ) )
         for j in range( 3 )                         ]

N = len( mfi_s )
# w = [ i / ( max( mfi_s ) ) for i in range( len( mfi_s ) ) ]
w = rfftfreq( N, 1/11. )

af_x = fft(mfi_b_x)
af_y = fft(mfi_b_y)
af_z = fft(mfi_b_z)

pf_x = [ degrees( angle(af_x[i] ) ) for i in range( len( mfi_s ) ) ]
pf_y = [ degrees( angle(af_y[i] ) ) for i in range( len( mfi_s ) ) ]
pf_z = [ degrees( angle(af_z[i] ) ) for i in range( len( mfi_s ) ) ]

saf_x = [af_x[i]**2 for i in range( len( f_x ) ) ]
saf_y = [af_y[i]**2 for i in range( len( f_x ) ) ]
saf_z = [af_z[i]**2 for i in range( len( f_x ) ) ]

sf_x = [f_x[i]**2 for i in range( len( f_x ) ) ]
sf_y = [f_y[i]**2 for i in range( len( f_x ) ) ]
sf_z = [f_z[i]**2 for i in range( len( f_x ) ) ]

#def model( t, bt, db, p ) :
#
#	return bt*t+db*cos( omega*t + p )
#
#( fitx, covarx ) = curve_fit( model, mfi_s, bx)
#( fity, covary ) = curve_fit( model, mfi_s, by)
#( fitz, covarz ) = curve_fit( model, mfi_s, bz)
#
#bx_m = [ fitx[0]*mfi_s[i] + fitx[1]*cos( omega * mfi_s[i] + fitx[2] )
#                                     for i in range( len( mfi_s ) ) ]
#by_m = [ fity[0]*mfi_s[i] + 0.16*cos( omega * mfi_s[i] + fity[2] )
#                                     for i in range( len( mfi_s ) ) ]
#bz_m = [ fity[0]*mfi_s[i] + 0.16*cos( omega * mfi_s[i] + fitz[2] )
#                                     for i in range( len( mfi_s ) ) ]

plt.close('all')

#plt.figure( )

#plt.plot( mfi_s, by, 'r' )
#plt.plot( mfi_s, by_m, 'b' )

#plt.figure( )
#plt.plot( mfi_s, bz, 'r' )
#plt.plot( mfi_s, bz_m, 'b' )
#plt.show( )

fig, axs = plt.subplots( 3, 2, sharex = 'col' )
fig.subplots_adjust(hspace=0)

axs[0,0].loglog( w[5:N//2], medfilt( abs( af_x[5:N//2] ), 5 ), 'r', label = 'f(x)' )
axs[0,0].set_ylim( 10**(-1), 2*10**1 )
axs[0,0].legend(loc="upper right")

axs[1,0].loglog( w[5:N//2], medfilt( abs( af_y[5:N//2] ), 5 ), 'b', label = 'f(y)' )
#plt.loglog( w[5:N//2], saf_y[5:N//2], label = 'saf_y' )
axs[1,0].set_ylim(10**(-1), 2*10**1)
axs[1,0].legend(loc="upper right")

axs[2,0].loglog( w[5:N//2], medfilt( abs( af_z[5:N//2] ), 5 ), 'k', label = 'f(z)' )
#plt.loglog( w[5:N//2], saf_z[5:N//2], label = 'saf_z' )
axs[2,0].set_ylim(10**(-1), 2*10**1)
axs[2,0].legend(loc="upper right")

axs[0,1].semilogx( w[5:N//2], angle( af_x[5:N//2] ) )
axs[1,1].semilogx( w[5:N//2], angle( af_y[5:N//2] ) )
axs[2,1].semilogx( w[5:N//2], angle( af_z[5:N//2] ) )

[ axs[j,0].axvline( 0.05, linestyle=':' ) for j in range( 3 ) ]
[ axs[j,1].axvline( 0.05, linestyle=':' ) for j in range( 3 ) ]

#plt.figure( )
#plt.plot( mfi_s[0:20], mfi_b_y[0:20], label = 'mfi_b_y' )
#plt.plot( mfi_s[0:20], mfi_b_z[0:20], label = 'mfi_b_z' )


plt.show( )

print 'Computation time = ','%.6f'% (time.time()-start), 'seconds.'
