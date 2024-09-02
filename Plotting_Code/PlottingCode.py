#import sys
# sys.path.append('../Scripts')
# from PostProcessingScripts import * 
# import pandas as pd 
# import string 
# just to make the cells appear wider:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import sys
import math
import seaborn as sns
import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import string
from astropy import units as u
from matplotlib import patches

class COMPASData(object):
    """ this class reads in datafile & gives you M1 and M2 and some other things """

    def __init__(self, path=None, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1):
        self.path                = path
        if (self.path is None):
            print("Just to double check you create instance of ClassCOMPAS without path/Data")
        elif not  os.path.isfile(path):
            raise ValueError("h5 file not found. Wrong path given?", "path given = %s"%path)
        elif os.path.isfile(path):
            self.h5file           = h5.File(path)
        # old code with fixed COMPAS h5 filename:
        #
        # elif not  os.path.isfile(path+'COMPASOutput.h5'):
        #     raise ValueError("h5 file not found. Wrong path given?", "path given = %s"%path)
        # elif os.path.isfile(path+'COMPASOutput.h5'):
        #     self.h5file           = h5.File(path+'COMPASOutput.h5')



        #Crucial values to be able to calculate MSSFR
        self.metallicityGrid     = None
        self.metallicitySystems  = None
        self.delayTimes          = None
        #Crucial values I need for selection effects
        self.mass1               = None
        self.mass2               = None

        self.m1zams               = None
        self.m2zams               = None
        self.separationzams       = None

        self.weight              = None  # //floor
        self.DCOmask             = None

        #Additional arrays that might be nice to store
        #to more quickly make some plots.
        #If you need more memory might help a tiny bit to not do
        self.lazyData            = lazyData
        self.pessimictic_allDCO  = None  #which of all DCO are pessimistic
        self.pessimictic_subSet  = None  #which of the systems calculated are pessimistic
        self.mChirp              = None
        self.q                   = None
        self.eccentricityDCOFormation = None
        self.separationDCOFormation =None
        self.tc                  = None
        self.seeds               = None
        self.systemicVelocity    = None

        self.Mlower              = Mlower
        self.Mupper              = Mupper
        self.binaryFraction      = binaryFraction
        self.totalMassEvolvedPerZ = None
        
#         if self.path is not None:
#            self.setGridAndMassEvolved()        

#         print("Remember to  self.setCOMPASDCOmask() and self.setCOMPASData()")

    
    def setCOMPASDCOmask(self, types='BBH', withinHubbleTime=True, optimistic=False):
            #We do not want all the formation channels just the ones that form BBHs
            fDCO    = self.h5file['doubleCompactObjects']
            if types == 'BBH':
                maskTypes = (fDCO['stellarType1'][...].squeeze() == 14) &\
                            (fDCO['stellarType2'][...].squeeze() == 14)
            elif types == 'BNS':
                maskTypes = (fDCO['stellarType1'][...].squeeze() == 13) &\
                            (fDCO['stellarType2'][...].squeeze() == 13)
            elif types == 'BHNS':
                maskTypes = ((fDCO['stellarType1'][...].squeeze() == 14) &\
                            (fDCO['stellarType2'][...].squeeze() == 13)) |\
                            ((fDCO['stellarType1'][...].squeeze() == 13) &\
                            (fDCO['stellarType2'][...].squeeze() == 14))
            elif types == 'ALL':
                maskTypes = ((fDCO['stellarType1'][...].squeeze() == 14) |\
                            (fDCO['stellarType1'][...].squeeze() == 13)) &\
                            ((fDCO['stellarType2'][...].squeeze() == 13) |\
                            (fDCO['stellarType2'][...].squeeze() == 14))
            else:
                raise ValueError('types=%s not in BBH, BNS, BHNS' %(types))         
            if withinHubbleTime == True:
                maskHubble = (fDCO['mergesInHubbleTimeFlag'][...].squeeze()==True)
            else:
                #Array where all are true
                maskHubble = np.ones(len(fDCO['mergesInHubbleTimeFlag'][...].squeeze()), dtype=bool)
                              
            if optimistic == True:
                #we do not care about the optimistic flag (both False and True allowed)
                #Array where all are true
                maskOptimistic = np.ones(len(fDCO['optimisticCEFlag'][...].squeeze()), dtype=bool)
            else:
                #optimistic scenario not allowed (pessimistic) hence the flag must be false
                #This removes systems with CEE from HG donors (no core envelope separation)
                maskOptimistic = fDCO['optimisticCEFlag'][...].squeeze() == False
                              
            #we never want in first timestep after CEE, because 
            #we define it as a system that should not have survived the CEE
            maskNoRLOFafterCEE =  (fDCO['RLOFSecondaryAfterCEE'][...].squeeze()==False)
                              

            self.DCOmask = maskTypes & maskHubble & maskOptimistic & maskNoRLOFafterCEE

#     def setGridAndMassEvolved(self):
    
#         #The COMPAS simulation does not evolve all stars 
#         #give me the correction factor for the total mass evolved
#         #I assume each metallicity has the same limits, and does correction
#         #factor, but the total mass evolved might be different.
#         #This does not change when we change types and other masks this is 
#         #general to the entire simulation so calculate once
#         _, self.totalMassEvolvedPerZ =\
#         MPZ.totalMassEvolvedPerZ(pathCOMPASh5=self.path , Mlower=self.Mlower, \
#                                  Mupper=self.Mupper, binaryFraction=self.binaryFraction)
#         #Want to recover entire metallicity grid, assume that every metallicity
#         #evolved shows in all systems again should not change within same run
#         #so dont redo if we reset the data
#         metallicities = self.h5file['systems']['Metallicity1'][...].squeeze()
#         self.metallicityGrid     = np.unique(metallicities)


    def setCOMPASData(self):
        fDCO      = self.h5file['doubleCompactObjects']
        #Stuff I need for cosmological integral
        

        #sorry not the prettiest line is a boolean slice of seeds
        #this only works because seeds in systems file and DCO file are printed
        #in same order

        #TODO more elegant slicing not unpack all and then mask 
        #something like f[M1][mask][...] but the the way I setup DCO mask needs different shape
        #metallicities = metallicities[np.in1d(self.h5file['systems']['SEED'][...].squeeze(),\
        #                        fDCO['seed'][...].squeeze()[self.DCOmask])]
        
        self.metallicitySystems  = fDCO['Metallicity1'][...].squeeze()[self.DCOmask]
        self.delayTimes          = fDCO['tform'][...].squeeze()[self.DCOmask] + \
                                   fDCO['tc'][...].squeeze()[self.DCOmask]
        self.tc                  = fDCO['tc'][...].squeeze()[self.DCOmask]
        self.mass1               = fDCO['M1'][...].squeeze()[self.DCOmask]
        self.mass2               = fDCO['M2'][...].squeeze()[self.DCOmask]
        self.weight              = fDCO['weight'][...].squeeze()[self.DCOmask] # //floor
        self.m1zams              = fDCO['M1ZAMS'][...].squeeze()[self.DCOmask]
        self.m2zams              = fDCO['M2ZAMS'][...].squeeze()[self.DCOmask]
        self.separationzams      = fDCO['separationInitial'][...].squeeze()[self.DCOmask]


        

        

        #Stuff of data I dont need for integral
        #but I might be to laze to read in myself
        #and often use. Might turn it of for memory efficiency
        if self.lazyData:
            self.pessimictic_allDCO  = fDCO['optimisticCEFlag'][...].squeeze()==False
            self.pessimictic_subSet  = fDCO['optimisticCEFlag'][...].squeeze()[self.DCOmask]==False
            self.q                   = np.divide(self.mass2, self.mass1)
            boolq                    = self.mass2 > self.mass1
            self.q[boolq]            = np.divide(self.mass1[boolq], self.mass2[boolq])
            self.mChirp = np.divide((np.multiply(self.mass2, self.mass1)**(3./5.) ),\
                                           (np.add(self.mass2, self.mass1)**(1./5.)))

            self.separationDCOFormation= fDCO['separationDCOFormation'][...].squeeze()[self.DCOmask]
            self.eccentricityDCOFormation= fDCO['eccentricityDCOFormation'][...].squeeze()[self.DCOmask]
            self.seeds               = fDCO['seed'][...].squeeze()[self.DCOmask]
            self.Hubble              = fDCO['mergesInHubbleTimeFlag'][...].squeeze()[self.DCOmask]

            fSN     = self.h5file['supernovae']
            seedsSN = fSN ['randomSeed'][...].squeeze()
            bools = np.in1d(seedsSN, self.seeds)
            self.systemicVelocity    = fSN['systemicVelocity'][...].squeeze()[bools]


#     def setGridAndMassEvolved(self):

#         #The COMPAS simulation does not evolve all stars 
#         #give me the correction factor for the total mass evolved
#         #I assume each metallicity has the same limits, and does correction
#         #factor, but the total mass evolved might be different.
#         #This does not change when we change types and other masks this is 
#         #general to the entire simulation so calculate once
#         _, self.totalMassEvolvedPerZ =\
#         MPZ.totalMassEvolvedPerZ(pathCOMPASh5=self.path , Mlower=self.Mlower, \
#                                  Mupper=self.Mupper, binaryFraction=self.binaryFraction)
#         #Want to recover entire metallicity grid, assume that every metallicity
#         #evolved shows in all systems again should not change within same run
#         #so dont redo if we reset the data
#         metallicities = self.h5file['systems']['Metallicity1'][...].squeeze()
#         self.metallicityGrid     = np.unique(metallicities)

# to obtain properties of ALL binaries simulated, do this:

DCOtype_BHNS = 'BHNS'   # You can change this line to 'BBH', 'BHNS' 'BNS', or 'ALL' (All DCOs)  # change this line! 

DCOtype_BNS = 'BNS'


# add path to where the COMPASOutput.h5 file is stored. 
# For you the part '/Volumes/Andromeda/DATA/AllDCO_bugfix/fiducial/' is probably different
path_BHNS = r"C:\Users\cauar\OneDrive\Área de Trabalho\Data\COMPASCompactOutput_"+ DCOtype_BHNS +'_A.h5' # change this line! 

path_BNS = r"C:\Users\cauar\OneDrive\Área de Trabalho\Data\COMPASCompactOutput_"+ DCOtype_BNS +'_A.h5'

fdata_BHNS = h5.File(path_BHNS)
fdata_BNS = h5.File(path_BNS)

fSN   = h5.File(path_BNS)
# shows the different files within the hdf5 folder 

fBH = h5.File(path_BHNS)['formationChannels']

print(fBH)
print('this might take a little while, particularly if you are using the BBH')


# This code below gets the COMPAS data and only the systems that are DCOs 
Data_BHNS            = COMPASData(path=path_BHNS, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
Data_BHNS .setCOMPASDCOmask(types=DCOtype_BHNS,  withinHubbleTime=True, optimistic=False)
Data_BHNS .setCOMPASData()

Data_BNS            = COMPASData(path=path_BNS, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
Data_BNS .setCOMPASDCOmask(types=DCOtype_BNS,  withinHubbleTime=True, optimistic=False)
Data_BNS .setCOMPASData()
# SeedsHubble    = Data.seeds[Data.Hubble==True]

# to obtain the properties of the selected DCOtype you simply do this:
M1_BHNS = Data_BHNS .mass1
M2_BHNS = Data_BHNS .mass2
weights_BHNS = Data_BHNS .weight

M1_BNS = Data_BNS .mass1
M2_BNS = Data_BNS .mass2
weights_BNS = Data_BNS .weight

Systemic_Velocity_BHNS = Data_BHNS .systemicVelocity
Delay_time_BHNS = Data_BHNS .tc
Eccentricities_BHNS = Data_BHNS .eccentricityDCOFormation
Semi_major_axis_BHNS = Data_BHNS .separationDCOFormation * 215.032

Systemic_Velocity_BNS = Data_BNS .systemicVelocity
Delay_time_BNS = Data_BNS .tc
Eccentricities_BNS = Data_BNS .eccentricityDCOFormation
Semi_major_axis_BNS= Data_BNS .separationDCOFormation * 215.032

DCOseeds_BHNS = Data_BHNS .seeds
seedsSN_BHNS = fdata_BHNS['supernovae']['randomSeed'][...].squeeze()
# get only SN seeds for DCOs 
maskSNdco_BHNS = np.in1d(seedsSN_BHNS,  DCOseeds_BHNS) 
Systemic_Velocities_of_Interest_BHNS = Systemic_Velocity_BHNS[maskSNdco_BHNS][1::2]

DCOseeds_BNS = Data_BNS .seeds
seedsSN_BNS = fdata_BNS['supernovae']['randomSeed'][...].squeeze()
# get only SN seeds for DCOs 
maskSNdco_BNS = np.in1d(seedsSN_BNS,  DCOseeds_BNS) 
#Systemic_Velocities_of_Interest_BNS = Systemic_Velocity_BNS[maskSNdco_BNS][1::2]

seeds_masked = seedsSN_BNS[maskSNdco_BNS]
print(len(Systemic_Velocity_BNS))

mask = Data_BHNS.metallicitySystems < (0.0142/5)
array = np.unique(Data_BHNS.metallicitySystems[mask])
np.set_printoptions(threshold = sys.maxsize)
print(len(array))
"""Systemic_Velocities_of_Interest_BNS = np.asarray([])
for seed in np.unique(seeds_masked):
    mask = np.in1d(seeds_masked, seed)
    systemic_velocities_masked = Systemic_Velocity_BNS[mask]
    Systemic_Velocities_of_Interest_BNS = np.append(Systemic_Velocities_of_Interest_BNS, systemic_velocities_masked[-1])
    
print(len(Systemic_Velocities_of_Interest_BNS))

def calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc):
    
    Escape_Velocity = math.sqrt(2*6.67*(10**-11)*galaxy_mass_solar_masses*1.989*(10**30)/(virial_radius_kpc * 3.08 * (10**19)))/1000
    
    return Escape_Velocity
    
def mask_Candidates(DCO_type, galaxy_mass_solar_masses, virial_radius_kpc):
    
    if DCO_type == "BHNS":

        Escape_Velocity = calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc)
        maskCandidates_BHNS = ((Delay_time_BHNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS * Delay_time_BHNS * 3.2 * (10**13) < 0.1 * virial_radius_kpc * 3.08 * (10**16))) | ((Delay_time_BHNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS < Escape_Velocity))
        seeds_candidates_BHNS = fdata_BHNS['doubleCompactObjects']['seed'][...].squeeze()[maskCandidates_BHNS]
        mask_seeds_candidates_BHNS = np.in1d(DCOseeds_BHNS, seeds_candidates_BHNS)
        
        percentage = round(((len(seeds_candidates_BHNS) / len(mask_seeds_candidates_BHNS))*100), 2)
        
        print("The percentage of candidate BHNS is: " + str(percentage) + "%")
        
        return mask_seeds_candidates_BHNS
        
    elif DCO_type == "BNS":
        
        Escape_Velocity = calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc)
        maskCandidates_BNS = ((Delay_time_BNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BNS * Delay_time_BNS* 3.2 * (10**13) < 0.1 * virial_radius_kpc * 3.08 * (10**16))) | ((Delay_time_BNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BNS < Escape_Velocity))
        seeds_candidates_BNS = fdata_BNS['doubleCompactObjects']['seed'][...].squeeze()[maskCandidates_BNS]
        mask_seeds_candidates_BNS = np.in1d(DCOseeds_BNS, seeds_candidates_BNS)
        
        percentage = round(((len(seeds_candidates_BNS) / len(mask_seeds_candidates_BNS))*100), 2)
        
        print("The percentage of candidate BNS is: " + str(percentage) + "%")
        
        return mask_seeds_candidates_BNS"""
    
"""mask_candidate_BHNS = mask_Candidates(DCO_type = "BHNS", galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6)

mask_candidate_BHNS_2 = mask_Candidates(DCO_type = "BHNS", galaxy_mass_solar_masses = (10**8),  virial_radius_kpc = 1.3)

mask_candidate_BNS = mask_Candidates(DCO_type = "BNS", galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6)"""

#### MAIN PLOT CODE: 

#### JUST BHNS:

"""metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
metallicitySystems_BHNS_2 = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS_2])

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
WEIGHTS_BHNS_2 = weights_BHNS[mask_candidate_BHNS_2]

metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)
metallicity_2, frequency_2 = np.unique(metallicitySystems_BHNS_2, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)

weights_sum_BHNS_2 = []

for x in np.asarray(metallicity_2):
    mask_metallicitySystems_BHNS_2 = np.in1d(metallicitySystems_BHNS_2, np.asarray(x))
    WEIGHTS_BHNS_masked_2 = WEIGHTS_BHNS_2[mask_metallicitySystems_BHNS_2] 
    weights_sum_BHNS_2 = np.append(weights_sum_BHNS_2, np.sum(WEIGHTS_BHNS_masked_2))

rates_2 = np.divide(weights_sum_BHNS_2, 77598320)

Max_Reticulum_II_Mass = 2.6*(10**4)
Min_Reticulum_II_Mass = 10.4*(10**4)

Max_Tucana_III_Mass = 4.6*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'
  
fig, ax = plt.subplots(figsize=(10,6))
plt.grid()
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='green', alpha=0.75, lw=0, label = "Reticulum II Progenitor Stellar Mass")
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.75, lw=0, label = "Tucana III Progenitor Stellar Mass")
plt.yscale('log')


n=0
for rate in rates_2:
    x=metallicity[n]
    rect = patches.Rectangle(xy=(x, rate), width=0.01, height=rate + rates[n])
    ax.add_patch(rect)
    n=n+1

plt.scatter(np.asarray(metallicity),rates, s=16, marker = '.', label = "Candidate BHNS")
plt.scatter(np.asarray(metallicity),(rates*(10**0.7)), s=16, c='navy', marker = '.')
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.legend(markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

#### BHNS vs BNS COMPARISON: 

"""metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
print(Data_BHNS.metallicitySystems)

metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)

metallicitySystems_BNS = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS])
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]
print(Data_BNS.metallicitySystems)

metallicity_BNS, frequency_BNS = np.unique(metallicitySystems_BNS, return_counts=True)

weights_sum_BNS = []

for x in np.asarray(metallicity_BNS):
    mask_metallicitySystems_BNS = np.in1d(metallicitySystems_BNS, np.asarray(x))
    WEIGHTS_BNS_masked = WEIGHTS_BNS[mask_metallicitySystems_BNS] 
    weights_sum_BNS = np.append(weights_sum_BNS, np.sum(WEIGHTS_BNS_masked))

rates_BNS = np.divide(weights_sum_BNS, 77598320)

Max_Reticulum_II_Mass = 2.6*(10**4)
Min_Reticulum_II_Mass = 10.4*(10**4)

Max_Tucana_III_Mass = 4.6*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

plt.figure(figsize=(10,6))
plt.grid()
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='green', alpha=0.75, lw=0, label = "Reticulum II Progenitor Stellar Mass")
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.75, lw=0, label = "Tucana III Progenitor Stellar Mass")
plt.scatter(np.asarray(metallicity),rates, s=16, marker = '.', label = "Candidate BHNS")
plt.scatter(np.asarray(metallicity),(rates*(10**0.7)), s=16, c='navy', marker = '.')
plt.scatter(np.asarray(metallicity_BNS),rates_BNS, s=16, c='red', marker = '.', label = "Candidate BNS")
plt.scatter(np.asarray(metallicity_BNS),(rates_BNS*(10**0.7)), s=16, c='darkred', marker = '.')
plt.yscale('log')
plt.xlabel(r'$ log_{\rm{10}}\mathcal{M}\ $')
plt.ylabel(r'$ \mathcal{R}\ $')
plt.legend(markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

#### FUNCTION CODE: 

#RATES AS A FUNCTION OF METALLICITY - ALL BHNS: 

"""metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems)
WEIGHTS_BHNS = weights_BHNS
print(Data_BHNS.metallicitySystems)

metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)

plt.figure(figsize=(10,6))
plt.plot(np.asarray(metallicity),rates)
plt.yscale('log')
plt.grid()
plt.xlabel(r'$ log_{\rm{10}}\mathcal{M}\ $')
plt.ylabel(r'$ \mathcal{R}\ $')
plt.legend(["ALL BHNS" , "Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

#RATES AS A FUNCTION OF METALLICITY - Candidate BHNS: 

"""metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
print(Data_BHNS.metallicitySystems)

metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)

plt.figure(figsize=(10,6))
plt.plot(np.asarray(metallicity),rates)
plt.yscale('log')
plt.grid()
plt.xlabel(r'$ log_{\rm{10}}\mathcal{M}\ $')
plt.ylabel(r'$ \mathcal{R}\ $')
plt.legend(["Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

#### BHNS and BNS comparison historigrams with rates: 

## TO MAKE COMPARISON OF METALLICITY OF CANDIDATES WITH RATES PLOT:

"""metallicitySystems_BNS = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS])
WEIGHTS_BNS = np.divide(weights_BNS[mask_candidate_BNS], 77598320)

metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = np.divide(weights_BHNS[mask_candidate_BHNS], 77598320)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

plt.figure(figsize=(10,6))
plt.grid()
plt.hist(metallicitySystems_BHNS, bins=100, weights=WEIGHTS_BHNS, color='teal', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BHNS")
plt.hist(metallicitySystems_BNS, bins=100, weights=WEIGHTS_BNS, color='gold', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BNS")
plt.legend()
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.show()"""

################################################

## TO MAKE COMPARISON OF SYSTEMIC VELOCITY OF CANDIDATES WITH RATES PLOT:

"""candidate_systemic_velocities_BHNS = np.log10(Systemic_Velocities_of_Interest_BHNS[mask_candidate_BHNS])
candidate_systemic_velocities_BNS = np.log10(Systemic_Velocities_of_Interest_BNS[mask_candidate_BNS])

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]/77598320
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]/77598320

nameX = r'$\log_{{\rm{10}}}v_{\rm{sys}} \ \ [Km/s] $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'


plt.grid()
plt.hist(candidate_systemic_velocities_BHNS, bins=100, weights=WEIGHTS_BHNS, color='teal', histtype='stepfilled', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BHNS")
plt.hist(candidate_systemic_velocities_BNS, bins=100, weights=WEIGHTS_BNS, color='gold', histtype='stepfilled', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BNS")
plt.legend()
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.show()"""

################################################

## TO MAKE COMPARISON OF SYSTEMIC VELOCITY OF CANDIDATES WITH RATES PLOT:

"""delayTimes_BHNS = np.log10(Delay_time_BHNS[mask_candidate_BHNS])
delayTimes_BNS = np.log10(Delay_time_BNS[mask_candidate_BNS])

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]/77598320
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]/77598320

nameX = r'$\log_{{\rm{10}}}D_{\rm{time}} \ \ [Myr] $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

plt.grid()
plt.hist(delayTimes_BHNS, bins=100, weights=WEIGHTS_BHNS, color='teal', histtype='stepfilled', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BHNS")
plt.hist(delayTimes_BNS, bins=100, weights=WEIGHTS_BNS, color='gold', histtype='stepfilled', edgecolor='black', linewidth=1.5, alpha=0.7, label="Candidate BNS")
plt.legend()
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.show()"""


################################################

#### BHNS and BNS comparison historigrams: 

## TO MAKE COMPARISON OF METALLICITY OF CANDIDATES PLOT:

"""metallicitySystems_BNS = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS])
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]

metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]

plt.figure(figsize=(10,6)) 
plt.hist(metallicitySystems_BHNS, bins=100, weights=WEIGHTS_BHNS, color='teal', alpha=0.6, edgecolor='black', label="Candidate BHNS")
plt.hist(metallicitySystems_BNS, bins=100, weights=WEIGHTS_BNS, color='gold', alpha=0.6, edgecolor='black', label="Candidate BNS")
plt.legend()
plt.grid()
plt.xlabel('metallicity of candidates')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE COMPARISON OF DELAY TIMES OF CANDIDATES PLOT:

"""candidate_semi_major_axis_BHNS = Semi_major_axis_BHNS[mask_candidate_BHNS]
candidate_semi_major_axis_BNS = Semi_major_axis_BNS[mask_candidate_BNS]

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]

plt.hist(candidate_semi_major_axis_BHNS, bins=100, weights=WEIGHTS_BHNS, color='violet', alpha=1, edgecolor='red', label="Candidate BHNS")
plt.hist(candidate_semi_major_axis_BNS, bins=100, weights=WEIGHTS_BNS, color='springgreen', alpha=0.5, edgecolor='blue', label="Candidate BNS")
plt.xlabel('Semi-major axis of candidates')
plt.ylabel('weighted rate in COMPAS simulation')
plt.xscale('log')
plt.show()"""

################################################

## TO MAKE COMPARISON OF DELAY TIMES OF CANDIDATES PLOT:

"""delayTimes_BHNS = np.log10(Delay_time_BHNS[mask_candidate_BHNS])
delayTimes_BNS = np.log10(Delay_time_BNS[mask_candidate_BNS])

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]

plt.hist(delayTimes_BHNS, bins=100, weights=WEIGHTS_BHNS, color='teal', alpha=0.6, edgecolor='black', label="Candidate BHNS")
plt.hist(delayTimes_BNS, bins=100, weights=WEIGHTS_BNS, color='gold', alpha=0.6, edgecolor='black', label="Candidate BNS")
plt.legend()
plt.xlabel('Delay time of candidates [Myr] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.grid()
plt.show()"""

################################################

## TO MAKE COMPARISON OF SYSTEMIC VELOCITY OF CANDIDATES PLOT:

"""candidate_systemic_velocities_BHNS = Systemic_Velocities_of_Interest_BHNS[mask_candidate_BHNS]
candidate_systemic_velocities_BNS = Systemic_Velocities_of_Interest_BNS[mask_candidate_BNS]

WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]

plt.hist(candidate_systemic_velocities_BHNS, bins=100, weights=WEIGHTS_BHNS, color='violet', alpha=1, edgecolor='red', label="Candidate BHNS")
plt.hist(candidate_systemic_velocities_BNS, bins=100, weights=WEIGHTS_BNS, color='springgreen', alpha=0.5, edgecolor='blue', label="Candidate BNS")
plt.legend()
plt.xlabel('Systemic Velocity of candidates [Km/s] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

#### TO BLACK HOLE - NEUTRON STAR SYSTEMS: 

## TO MAKE METALLICITY OF CANDIDATES PLOT:

"""metallicitySystems = Data.metallicitySystems[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(metallicitySystems, bins=100, weights=WEIGHTS)
plt.xlabel('metallicity of BHNS candidates ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE METALLICITY VS DELAY TIMES OF CANDIDATES PLOT:

"""metallicitySystems = Data.metallicitySystems[final_mask_candidates]
delayTimes = Delay_time[final_mask_candidates]

plt.scatter(metallicitySystems, delayTimes, s=4, marker = '.')
plt.xlabel('Metallicity BHNS candidates')
plt.ylabel('Delay Time [Myr]')
plt.xscale('log')
plt.show()"""

################################################

## TO MAKE SEPARATION OF CANDIDATES PLOT:

"""candidate_semi_major_axis_BHNS = Semi_major_axis_BHNS[mask_candidate_BHNS]
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]

plt.hist(candidate_semi_major_axis_BHNS, bins=100, weights=WEIGHTS_BHNS)
plt.xlabel('Semi-major axis of BHNS candidates')
plt.ylabel('weighted rate in COMPAS simulation')
plt.xscale('log')
plt.show()"""

################################################

## TO MAKE SYSTEMIC VELOCITY OF CANDIDATES PLOT:

"""candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(candidate_systemic_velocities, bins=100, weights=WEIGHTS)
plt.xlabel('Systemic Velocity of BHNS candidates [Km/s] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE DELAY TIMES OF CANDIDATES PLOT:

"""delayTimes = Delay_time[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(delayTimes, bins=100, weights=WEIGHTS)
plt.xlabel('Delay time of BHNS candidates [Myr] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE MASS OF CANDIDATES PLOT:

"""M1_candidates = M1[final_mask_candidates]
M2_candidates = M2[final_mask_candidates]

def obtainM1BHandM2BHassymetric(m1, m2):
    m1bh, m2bh = np.zeros_like(m1), np.zeros_like(m1)
    maskm1heavier = ( m1 >= m2)
    maskm2heavier = (m1 < m2)
    
    m1bh[maskm1heavier] = m1[maskm1heavier] 
    m1bh[maskm2heavier] = m2[maskm2heavier]
    m2bh[maskm1heavier] = m2[maskm1heavier]
    m2bh[maskm2heavier] = m1[maskm2heavier]
    
    return m1bh, m2bh # m1bh has all the heaviest systems

M_most_massive, M_least_massive = obtainM1BHandM2BHassymetric(m1=M1_candidates, m2=M2_candidates)


plt.scatter(M_most_massive, M_least_massive, s=4, marker = '.')
plt.xlabel('Mass of Candidate Black Holes [Msun]') #assumes that the most massive compact object in the BHNS system is the black hole
plt.ylabel('Mass of Candidate Neutron Stars [Msun]')
plt.show()"""

################################################

## TO MAKE SEMI-MAJOR AXIS VS ECCENTRICITY PLOT WITH CANDIDATES MARKED:

#candidate_semi_major_axis = Semi_major_axis[final_mask_candidates]
#candidate_eccentricities = Eccentricities[final_mask_candidates]

#plt.scatter(Semi_major_axis, Eccentricities, s=4, c='gray', marker = '.')
#plt.scatter(candidate_semi_major_axis, candidate_eccentricities, s=4, c='magenta', marker = '.')
#plt.xscale('log')
#plt.xlabel(r'(a)$[R_{\odot}]$')
#plt.ylabel('Eccentricity')
#plt.legend(["ALL BHNS" , "Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
#plt.show()

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH CANDIDATES MARKED:

#candidate_delay_times = Delay_time[final_mask_candidates]
#candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
      
#plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c='gray', marker = '.')
#plt.scatter(candidate_systemic_velocities, candidate_delay_times, s =4, c='magenta', marker = '.')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(r'$v_{CM}$$[km/s]$')
#plt.ylabel(r'$t_{merge}$$[Myr]$')
#plt.legend(["ALL BHNS" , "Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
#plt.show()

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH CANDIDATES, ESCAPE VELOCITY, AND METALLICITY MARKED:

"""candidate_delay_times = Delay_time[final_mask_candidates]
candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
EscapeVelocity = calculate_escape_velocity(galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6)
metallicitySystems = Data.metallicitySystems[final_mask_candidates]

plt.figure(figsize=(10,6))      
plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c='gray', marker = '.')
plt.scatter(candidate_systemic_velocities, candidate_delay_times, s =4, c=metallicitySystems, marker = '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_{CM}$$[km/s]$')
plt.ylabel(r'$t_{merge}$$[Myr]$')
plt.plot(np.array([EscapeVelocity, EscapeVelocity]),np.array([0, Delay_time.max()])) 
plt.plot(np.array([0, Systemic_Velocities_of_Interest.max()]),np.array([1000, 1000]))   
 
plt.colorbar(label='metallicity')
plt.legend(["ALL BHNS" , "Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

## TO SEE THE DELAY TIME DISTRIBUTION

#plt.hist(Delay_time, bins=100, weights=weights)
#plt.xlabel('Delay time [Myr] ')
#plt.ylabel('weighted rate in COMPAS simulation')
#plt.show()

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH ECCENTRICITY MARKED:

#plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c=Eccentricities, marker = '.')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(r'$v_{CM}$$[km/s]$')
#plt.ylabel(r'$t_{merge}$$[Myr]$')
#plt.colorbar(label='eccentricity')
#plt.ylim(10**-12, 10**3)
#plt.show()

################################################

## TO MAKE SEMI-MAJOR AXIS VS ECCENTRICITY PLOT:

#plt.scatter(Semi_major_axis, Eccentricities, s=4, marker = '.')
#plt.xscale('log')
#plt.xlabel(r'(a)$[R_{\odot}]$')
#plt.ylabel('Eccentricity')
#plt.show()

""" shortcut """

################################################################################################

#### TO BINARY NEUTRON STAR SYSTEMS:

## TO MAKE METALLICITY OF CANDIDATES PLOT:

"""metallicitySystems = Data.metallicitySystems[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(metallicitySystems, bins=100, weights=WEIGHTS)
plt.xlabel('metallicity of BNS candidates ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE DELAY TIMES OF CANDIDATES PLOT:

"""delayTimes = Delay_time[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(delayTimes, bins=100, weights=WEIGHTS)
plt.xlabel('Delay time of BNS candidates [Myr] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE MASS OF CANDIDATES PLOT:

"""M1_candidates = M1[final_mask_candidates]
M2_candidates = M2[final_mask_candidates]

def obtainM1BHandM2BHassymetric(m1, m2):
    m1bh, m2bh = np.zeros_like(m1), np.zeros_like(m1)
    maskm1heavier = ( m1 >= m2)
    maskm2heavier = (m1 < m2)
    
    m1bh[maskm1heavier] = m1[maskm1heavier] 
    m1bh[maskm2heavier] = m2[maskm2heavier]
    m2bh[maskm1heavier] = m2[maskm1heavier]
    m2bh[maskm2heavier] = m1[maskm2heavier]
    
    return m1bh, m2bh # m1bh has all the heaviest systems

M_most_massive, M_least_massive = obtainM1BHandM2BHassymetric(m1=M1_candidates, m2=M2_candidates)


plt.scatter(M_most_massive, M_least_massive, s=4, marker = '.')
plt.xlabel('Most massive candidate[Msun]')
plt.ylabel('Least massive candidate[Msun]')
plt.show()"""

################################################

## TO MAKE SEMI-MAJOR AXIS VS ECCENTRICITY PLOT WITH CANDIDATES MARKED:

"""candidate_semi_major_axis = Semi_major_axis[final_mask_candidates]
candidate_eccentricities = Eccentricities[final_mask_candidates]

plt.scatter(Semi_major_axis, Eccentricities, s=4, c='gray', marker = '.')
plt.scatter(candidate_semi_major_axis, candidate_eccentricities, s=4, c='magenta', marker = '.')
plt.xscale('log')
plt.xlabel(r'(a)$[R_{\odot}]$')
plt.ylabel('Eccentricity')
plt.legend(["ALL BNS" , "Candidate BNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH CANDIDATES MARKED:

"""candidate_delay_times = Delay_time[final_mask_candidates]
candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
      
plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c='gray', marker = '.')
plt.scatter(candidate_systemic_velocities, candidate_delay_times, s =4, c='magenta', marker = '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_{CM}$$[km/s]$')
plt.ylabel(r'$t_{merge}$$[Myr]$')
plt.legend(["ALL BNS" , "Candidate BNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

## TO SEE THE DELAY TIME DISTRIBUTION

#plt.hist(Delay_time, bins=100, weights=weights)
#plt.xlabel('Delay time [Myr] ')
#plt.ylabel('weighted rate in COMPAS simulation')
#plt.show()

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH ECCENTRICITY MARKED:

"""plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c=Eccentricities, marker = '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_{CM}$$[km/s]$')
plt.ylabel(r'$t_{merge}$$[Myr]$')
plt.colorbar(label='eccentricity')
plt.show()"""

################################################

## TO MAKE SEMI-MAJOR AXIS VS ECCENTRICITY PLOT:

#plt.scatter(Semi_major_axis, Eccentricities, s=4, marker = '.')
#plt.xscale('log')
#plt.xlabel(r'(a)$[R_{\odot}]$')
#plt.ylabel('Eccentricity')
#plt.show()

################################################

## TO MAKE METALLICITY VS DELAY TIMES OF CANDIDATES PLOT:

"""metallicitySystems = Data.metallicitySystems[final_mask_candidates]
delayTimes = Delay_time[final_mask_candidates]

plt.scatter(metallicitySystems, delayTimes, s=4, marker = '.')
plt.xlabel('Metallicity BNS candidates')
plt.ylabel('Delay Time [Myr]')
plt.xscale('log')
plt.show()"""

################################################

## TO MAKE SYSTEMIC VELOCITY OF CANDIDATES PLOT:

"""candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(candidate_systemic_velocities, bins=100, weights=WEIGHTS)
plt.xlabel('Systemic Velocity of BNS candidates [Km/s] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE SEPARATION OF CANDIDATES PLOT:

"""candidate_semi_major_axis = Semi_major_axis[final_mask_candidates]
WEIGHTS = weights[final_mask_candidates]

plt.hist(candidate_semi_major_axis, bins=100, weights=WEIGHTS)
plt.xlabel('Semi-major axis of BNS candidates [Km/s] ')
plt.ylabel('weighted rate in COMPAS simulation')
plt.show()"""

################################################

## TO MAKE SYSTEMIC_VELOCITY VS DELAY_TIME PLOT WITH CANDIDATES AND ESCAPE VELOCITY MARKED:

"""candidate_delay_times = Delay_time[final_mask_candidates]
candidate_systemic_velocities = Systemic_Velocities_of_Interest[final_mask_candidates]
EscapeVelocity = calculate_escape_velocity(galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6)
metallicitySystems = Data.metallicitySystems[final_mask_candidates]

plt.figure(figsize=(10,6))       
plt.scatter(Systemic_Velocities_of_Interest, Delay_time, s=4, c='gray', marker = '.')
plt.scatter(candidate_systemic_velocities, candidate_delay_times, s =4, c=metallicitySystems, marker = '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_{CM}$$[km/s]$')
plt.ylabel(r'$t_{merge}$$[Myr]$')
plt.plot(np.array([EscapeVelocity, EscapeVelocity]),np.array([0, Delay_time.max()])) 
plt.plot(np.array([0, Systemic_Velocities_of_Interest.max()]),np.array([1000, 1000]))   
plt.colorbar(label='metallicity')
plt.legend(["ALL BNS" , "Candidate BNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

