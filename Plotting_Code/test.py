# first we will import some of the packages that we will use 
import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt

# we will use astropy for some useful constants and units 
from astropy import units as u
from astropy import constants as const
from matplotlib.ticker import (FormatStrFormatter,
                               AutoMinorLocator)
from IPython.display import Image # to open images in Ipython

# add path to where the COMPASOutput.h5 file is stored. 
# For you the part '~/Downloads/' is probably different
path = r"C:\Users\cauar\OneDrive\Área de Trabalho\COMPAS_Output\COMPAS_Output.h5" # change this line! 

# the following line reads in the data 
fdata = h5.File(path, 'r')
#list(fdata.keys())  # print the different files within the hdf5 folder: 

#print(list(fdata.keys()))

#print(list(fdata['DoubleCompactObjects'].keys()))

#print()

#print(list(fdata['SystemParameters'].keys()))

#print()

#print(list(fdata['Supernovae'].keys()))

SEED_DCO = fdata['DoubleCompactObjects']["SEED"][...].squeeze()
#print(SEED_DCO)

# Method 2: Or even shorter: 

# all three groups contain the parameter "SEED", so we can use this to print the lengths
#for group in ['DoubleCompactObjects', 'SystemParameters', 'Supernovae']:
   # print('group %s has %s number of rows'%(group, fdata[group]["SEED"].len()))
   
# this is just a little function that we will use to make the plot more beautiful (bigger ticks, labels)
# However, you do not have to use this (just uncommoment "layoutAxes" everywhere)

def layoutAxes(ax, nameX='', nameY='', \
               labelSizeMajor = 10, fontsize = 25, second=False, labelpad=None, setMinor=True):
    """
    Tiny code to do the layout for axes in matplotlib
    """
    tickLengthMajor = 10
    tickLengthMinor = 5
    tickWidthMajor  = 1.5
    tickWidthMinor  = 1.5
    
    #rc('axes', linewidth=2)
    #label1 always refers to first axis not the twin 
    if not second:
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            #tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            #tick.label1.set_fontweight('bold')
    if second:
        for tick in ax.xaxis.get_major_ticks():
            tick.label2.set_fontsize(fontsize)
            #tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label2.set_fontsize(fontsize)
            #tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.tick_params(length=tickLengthMajor, width=tickWidthMajor, which='major')
    ax.tick_params(length=tickLengthMinor, width=tickWidthMinor, which='minor')
    ax.set_xlabel(nameX, fontsize=fontsize,labelpad=labelpad)#,fontweight='bold')
    ax.set_ylabel(nameY, fontsize=fontsize,labelpad=labelpad)#, fontweight='bold')    
    
    if setMinor==True:
        # add minor ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    return ax
fDCO      = fdata['DoubleCompactObjects']


M1 =  fDCO['Mass_1'][...].squeeze()  # mass in Msun of the compact object resulting from the *primary star*
M2 =  fDCO['Mass_2'][...].squeeze()  # mass in Msun of the compact object resulting from the *secondary star*


fsys      = fdata['SystemParameters']

def separation_to_period_circular_case(separation=10*u.AU, M1=1*u.M_sun, M2=1*u.M_sun):
    """calculate Period from separation
    separation is separation of the binary (needs to be given in astropy units)
    M1 and M2 are masses of the binary
    This is based on Kepler's law, using a circular orbit
    
    """
    G = const.G # [g cm s^2]
    

    mu = G*(M1+M2)
    period = 2*np.pi * np.sqrt(separation**3/mu)
    
    return period

CH_on_MS_1 =  fsys['CH_on_MS_1'][...].squeeze()  # mass in Msun of the compact object resulting from the primary
CH_on_MS_2 =  fsys['CH_on_MS_2'][...].squeeze()  # mass in Msun of the compact object resulting from the secondary
Stellar_TypeZAMS_1  =  fsys['Stellar_Type@ZAMS_1'][...].squeeze()  # mass in Msun of the compact object resulting from the primary
Stellar_TypeZAMS_2  =  fsys['Stellar_Type@ZAMS_2'][...].squeeze()  # mass in Msun of the compact object resulting from the secondary

# binaries that merge at birth as stars
Merger_At_Birth = fsys['Merger_At_Birth'][...].squeeze()

# SEED of the system Parameters (unique number corresponding to each binary)
SEED = fsys['SEED'][...].squeeze()  # mass in Msun of the compact object resulting from the secondary

# the CHE systems are then selected by systems that are CHE on ZAMS (stellar type 16) AND remain CHE on the MS (main sequence)
# in addition we do not want systems that Merged at Birth 
mask_CHE =  (CH_on_MS_1==1) & (CH_on_MS_2==1) & (Stellar_TypeZAMS_1==16) &  (Stellar_TypeZAMS_2==16) & (Merger_At_Birth==0)


# let's find the seed of the CHE systems: 
SEED_CHE = SEED[mask_CHE]

mask_M1isNS = (M1 <= 2.5) # M1 is a NS if mass is <= 2.5 Msun 
mask_M2isNS = (M2 <= 2.5) # M2 is a NS if mass is <= 2.5 Msun 
mask_BHBH = ((mask_M1isNS==0) & (mask_M2isNS==0))  # if true then the system is a BHBH

separation =  fDCO['Separation@DCO'][...].squeeze() # in AU 
Period = separation_to_period_circular_case(separation*u.au, M1*u.M_sun, M2*u.M_sun)
# the merger time is called the "coalescence time"
coalescence_time =  fDCO['Coalescence_Time'][...].squeeze() * u.Myr # Myr 
t_Hubble = 13.7 *u.Gyr
mask_tHubble = (coalescence_time < t_Hubble)

# this is the parameter that describes the Wolf Rayet factor f_WR that was used. 
WR_Multiplier = fsys['WR_Multiplier'][...].squeeze()

# mask BBHs that merge in a Hubble time 
mask_systemsOfInterest = (mask_BHBH==1) & (mask_tHubble==1)

# add the mask of systems that are CHE. Since the CHE mask is based on systemParameters we have 
# to match the systems from systemParameters that we want to mask with the DCO systems, we can do this using the SEED
# a system in systemParameters will have the same SEED in DoubleCompactObjects, if it exists in both
mask_DCO_that_are_CHE = np.in1d(SEED_DCO, SEED_CHE) 
mask_DCO_that_are_BBH_and_CHE = (mask_DCO_that_are_CHE ==1) & (mask_systemsOfInterest==1)

# we can mask for the f_WR = 0.2 factor that is used in Figure 6 of the paper. 
mask_fWR_02 = (WR_Multiplier==0.2)
SEED_fWR_02 = SEED[mask_fWR_02]
mask_DCO_that_are_fWR_02 = np.in1d(SEED_DCO, SEED_fWR_02)

# combine all the masks above
mask_DCO_that_are_BBH_and_CHE_and_fWR_02 = (mask_DCO_that_are_CHE ==1) & (mask_systemsOfInterest==1) & (mask_DCO_that_are_fWR_02==1)


mask_M1isNS = (M1 <= 2.5) # M1 is a NS if mass is <= 2.5 Msun 
mask_M2isNS = (M2 <= 2.5) # M2 is a NS if mass is <= 2.5 Msun 
mask_BHBH = ((mask_M1isNS==0) & (mask_M2isNS==0))  # if true then the system is a BHBH




M1ZAMS =  fsys['Mass@ZAMS_1'][...].squeeze()  # mass in Msun of the compact object resulting from the *primary star*
M2ZAMS =  fsys['Mass@ZAMS_2'][...].squeeze()  # mass in Msun of the compact object resulting from the *secondary star*


# the separation At ZAMS is given by 
separationZAMS = fsys['Separation@ZAMS'][()]
PeriodZAMS = separation_to_period_circular_case(separationZAMS*u.au, M1ZAMS*u.M_sun, M2ZAMS*u.M_sun)

# the merger time is called the "coalescence time"
coalescence_time =  fDCO['Coalescence_Time'][...].squeeze() * u.Myr # Myr 
t_Hubble = 13.7 *u.Gyr
mask_tHubble = (coalescence_time < t_Hubble)

# this is the parameter that describes the Wolf Rayet factor f_WR that was used. 
WR_Multiplier = fsys['WR_Multiplier'][...].squeeze()

# mask BBHs that merge in a Hubble time 
mask_systemsOfInterest = (mask_BHBH==1) & (mask_tHubble==1)

# add the mask of systems that are CHE. Since the CHE mask is based on systemParameters we have 
# to match the systems from systemParameters that we want to mask with the DCO systems, we can do this using the SEED
# a system in systemParameters will have the same SEED in DoubleCompactObjects, if it exists in both
mask_DCO_that_are_CHE = np.in1d(SEED_DCO, SEED_CHE) 
mask_DCO_that_are_BBH_and_CHE = (mask_DCO_that_are_CHE ==1) & (mask_systemsOfInterest==1)

# we can mask for the f_WR = 1 factor that is used in Figure 4 of the paper. 
mask_fWR_1 = (WR_Multiplier==1)
SEED_fWR_1 = SEED[mask_fWR_1]
mask_DCO_that_are_fWR_1 = np.in1d(SEED_DCO, SEED_fWR_1)

# combine all the masks above
mask_DCO_that_are_BBH_and_CHE_and_fWR_1 = (mask_DCO_that_are_CHE ==1) & (mask_systemsOfInterest==1) & (mask_DCO_that_are_fWR_1==1)

mask_Fig4 = np.in1d(SEED, SEED_DCO[mask_DCO_that_are_BBH_and_CHE_and_fWR_1])




# plot the systems 
f, ax= plt.subplots(1, 1,  figsize=(10,10)) 

plt.scatter((M1ZAMS+M2ZAMS)[mask_Fig4], PeriodZAMS[mask_Fig4].to(u.d))

xlabel = 'Total Mass at ZAMS [Msun]'
ylabel = 'Period at ZAMS [day]'
layoutAxes(ax=ax, nameX=xlabel,nameY=ylabel)
plt.xscale('log')        


plt.show()


print(len((M1+M2)[mask_DCO_that_are_BBH_and_CHE_and_fWR_02]))