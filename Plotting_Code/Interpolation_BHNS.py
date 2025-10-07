from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py as h5
import gc

#Quick fudge to make import from ../Scripts work
sys.path.append('../../Scripts')

#Custom imports
import ClassCOMPAS     as CC ###
# import ClassFormationChannels as FC 


# import script that has formation channel classification functions:
from PostProcessingScripts import * 
from ClassFormationChannels_5mainchannels import * 



from astropy import units as u
from astropy import constants as const

c= const.c.si
G = const.G.si

def timeToCoalescenceUsingInterpolation( a0,  e0,  m1,  m2):
    """/*
     Calculate the time to coalescence for a binary with arbitrary eccentricity using interpolation

     This is Equation 5.14 in Peters 1964 http:#journals.aps.org/pr/pdf/10.1103/PhysRev.136.B1224

     Parameters
     -----------
     a0 : 
        Initial semi-major axis in SI units
     e0 : 
        Initial eccentricity
     m1 :  
        Primary mass in SI units
     m2 : 
        Secondary mass in SI units

     Returns
     --------
     t_coalesce : 
        Time to coalescence in SI units (s)

     */
    """
    # Declare some variables
    tc       = 0.0;                                          # Time for a circular binary to coalesce
    beta     = 0.0;                                          # Beta constant (function of masses)
    tovertc  = 0.0;                                          # Result of interpolation of t/tc function


    # Calculate circular time to coalescence
    beta = calculateBeta(m1, m2);                                   # Masses should be in SI
    tc = a0*a0*a0*a0/(4.0*beta);                                    # Calculate time for a circular binary to merge

    # calculate t/tc using the interpolated function
    #tovertc = polynomialFitToToverTc(e0);                           # Ratio of inspiral time to that of circular system
    if((e0==0.0)or(e0==0)):
        return	tc;

    c0=a0*(1.0-e0*e0)*np.power(e0,-12.0/19.0)*np.power(1.0+(121.0*e0*e0/304.0), -870.0/2299.0);


    if(e0<0.01): 
        return c0*c0*c0*c0*np.power(e0,48.0/19.0)/(4.0*beta);


    if(e0>0.99):
        # Approximation of eq. 5.14 of Peters 1964, for high eccentricities
        return (768.0/425.0)*tc*np.power((1.0-(e0*e0)),3.5);

    # otherwise: numerically integrate
    summ=0;
    de=e0/10000;
    e=0
    while(e<e0):
        summ=summ+de*np.power(e,29.0/19.0)*np.power((1.0+(121.0/304.0)*e*e),1181.0/2299.0)/np.power((1-e*e),1.5);
        e=e+de

    return 12.0/19.0*c0*c0*c0*c0/beta*summ;



def calculateBeta( m1,  m2):
    """/*
     # Calculate the constant beta based on masses as defined in Equation 5.9 in Peters 1964 http:#journals.aps.org/pr/pdf/10.1103/PhysRev.136.B1224
     
     Parameters
     -----------
     m1 : float
        Primary mass in SI units
     m2 : float
        Secondary mass in SI units
     
     Returns
     --------
     beta : float
        Mass parameter beta
 
     */
     """
    M = m1 + m2;

    return (64.0/5.0)*G*G*G*m1*m2*M*np.power(c,-5.0);

def create_adaptive_separationArray(xstart,xend, Niter_a1, Niter_a2, Niter_a3):

    x_sep1 = np.logspace(-1.1, xstart, Niter_a1)
    x_sep2 = np.logspace(xstart*1.0001, xend, Niter_a2)
    x_sep3 = np.logspace(xend*1.0001, 3, Niter_a3)

    
    x_separation = np.concatenate((x_sep1, x_sep2))
    x_separation = np.concatenate((x_separation, x_sep3))
    
    x_separation = x_separation*u.R_sun # array of separations to try
    
    return x_separation
    




def obtain_a_and_e_for_fixed_mergerTime(x_separation, fixed_Tinsp, Niter_a, Niter_e):
    """ 
    
    fixed_Tinsp = wanted merger time in yrs
    
    """
    # constants
    M1 = 10*u.M_sun
    M2 = 1.4*u.M_sun
    c= const.c.si
    G = const.G.si
    ##
    if x_separation==None:
        x_separation = np.logspace(-5, 3, Niter_a)*u.R_sun # array of separations to try
    x_eccentricities = [] # list where we will append the solution of the eccentricities 

    emin =0
    for ind_a, sep in enumerate(x_separation):
        
        for ind_e, ecc in enumerate(np.linspace(emin, 1, Niter_e)):
            mergertime = timeToCoalescenceUsingInterpolation(a0=sep.to(u.meter), e0=ecc, m1=M1.to(u.kg),  m2=M2.to(u.kg)).to(u.year) # in years
            if (mergertime <= fixed_Tinsp) | (ecc==1):
                x_eccentricities.append(ecc)
                emin = ecc

                break 
    
    return x_eccentricities

Niter = 1000

a =10**(2)*u.R_sun
M1 = 10*u.M_sun
M2 = 1.4*u.M_sun

c= const.c.si
G = const.G.si
Niter_a, Niter_e = 20,20
t_Hubble = 13.7*u.Gyr
t_Hubble_inyears = (t_Hubble.to(u.year))
req_times_list_names = ['thousandYr' ,'tenMyr', 'oneGyr', 'tHubble' ]
BH_array = np.linspace(2.5, 20, Niter)


for ind_n, name_t in enumerate(req_times_list_names):

    TIME=name_t #'oneMyr'#'thousandYr'  #'tHubble'  #'tenMyr' #

    DEFAULT=False
    if DEFAULT==True:
        t_requested =  (1e6*u.year).to(u.year) #t_Hubble_inyears #(10*u.Myr).to(u.year) # in years
        
        
    if DEFAULT==True:
        x_separation = np.logspace(-1, 3, Niter_a)*u.R_sun # array of separations to try
        x_eccentricities = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation, fixed_Tinsp=t_requested, Niter_a=20, Niter_e=20)




    else:
        if TIME=='tHubble':
            xstart, xend = 0.8, 1.8
            x_separation_tHubble = create_adaptive_separationArray(xstart,xend, Niter_a1=10, Niter_a2=50, Niter_a3=30)
            x_separation=x_separation_tHubble

            x_eccentricities_tHubble = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation, fixed_Tinsp=t_Hubble_inyears, Niter_a=20, Niter_e=50)
            x_eccentricities=x_eccentricities_tHubble

        elif TIME=='oneGyr':
            t_requested = (1*u.Gyr).to(u.year)
            xstart, xend = np.log10(4), np.log10(50)

            x_separation_tGyr = create_adaptive_separationArray(xstart,xend, Niter_a1=10, Niter_a2=50, Niter_a3=30)
            x_separation_BHNS_1=x_separation_tGyr

            x_eccentricities_tGyr = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation_BHNS_1, fixed_Tinsp=t_requested, Niter_a=20, Niter_e=50)
            x_eccentricities_BHNS_1=x_eccentricities_tGyr

        elif TIME=='tenMyr':
            t_requested = (10*u.Myr).to(u.year)
            xstart, xend = np.log10(1), np.log10(20)


            x_separation_tTenMyr = create_adaptive_separationArray(xstart,xend, Niter_a1=10, Niter_a2=50, Niter_a3=30)
            x_separation_BHNS_2=x_separation_tTenMyr

            x_eccentricities_tTenMyr = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation_BHNS_2, fixed_Tinsp=t_requested, Niter_a=20, Niter_e=50)
            x_eccentricities_BHNS_2=x_eccentricities_tTenMyr        


        elif TIME=='oneMyr':
            t_requested = (1*u.Myr).to(u.year)
            xstart, xend = np.log10(.6), np.log10(4)


            x_separation_toneMyr = create_adaptive_separationArray(xstart,xend, Niter_a1=10, Niter_a2=50, Niter_a3=30)
            x_separation=x_separation_toneMyr

            x_eccentricities_toneMyr = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation, fixed_Tinsp=t_requested, Niter_a=20, Niter_e=50)
            x_eccentricities=x_eccentricities_toneMyr  



        elif TIME=='thousandYr':
            t_requested = (1000*u.year).to(u.year)
            xstart, xend = np.log10(.101), np.log10(.6)


            x_separation_thousandYr  = create_adaptive_separationArray(xstart,xend, Niter_a1=10, Niter_a2=50, Niter_a3=30)
            x_separation=x_separation_thousandYr 

            x_eccentricities_thousandYr  = obtain_a_and_e_for_fixed_mergerTime(x_separation=x_separation, fixed_Tinsp=t_requested, Niter_a=20, Niter_e=50)
            x_eccentricities=x_eccentricities_thousandYr        


        else:
            print('error time not known')

print(len(x_eccentricities_BHNS_2))

