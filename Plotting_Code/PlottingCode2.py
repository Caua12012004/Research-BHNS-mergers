#import sys
# sys.path.append('../Scripts')
# from PostProcessingScripts import * 
# import pandas as pd 
# import string 
# just to make the cells appear wider:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from PostProcessingScripts import layoutAxes
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

print(path_BHNS)
fdata_BHNS = h5.File(path_BHNS)

fdata_BNS = h5.File(path_BNS)

fSN   = h5.File(path_BHNS)['supernovae']
# shows the different files within the hdf5 folder 

print('this might take a little while, particularly if you are using the BBH')


# This code below gets the COMPAS data and only the systems that are DCOs 
Data_BHNS            = COMPASData(path=path_BHNS, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
Data_BHNS .setCOMPASDCOmask(types=DCOtype_BHNS,  withinHubbleTime=True, optimistic=True)
Data_BHNS .setCOMPASData()

Data_BNS            = COMPASData(path=path_BNS, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
Data_BNS .setCOMPASDCOmask(types=DCOtype_BNS,  withinHubbleTime=True, optimistic=False)
Data_BNS .setCOMPASData()
#SeedsHubble    = Data.seeds[Data.Hubble==True]

# to obtain the properties of the selected DCOtype you simply do this:
M1_BHNS = Data_BHNS .mass1
M2_BHNS = Data_BHNS .mass2
weights_BHNS = Data_BHNS .weight

print(len(M1_BHNS))

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
Systemic_Velocities_of_Interest_BHNS = Systemic_Velocity_BHNS[1::2]

print(len(DCOseeds_BHNS))
DCOseeds_BNS = Data_BNS .seeds
seedsSN_BNS = fdata_BNS['supernovae']['randomSeed'][...].squeeze()
# get only SN seeds for DCOs 
maskSNdco_BNS = np.in1d(seedsSN_BNS,  DCOseeds_BNS) 
Systemic_Velocities_of_Interest_BNS = Systemic_Velocity_BNS[maskSNdco_BNS][1::2]



def calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc):
    
    Escape_Velocity = math.sqrt(2*6.67*(10**-11)*galaxy_mass_solar_masses*1.989*(10**30)/(virial_radius_kpc * 3.08 * (10**19)))/1000
    
    return Escape_Velocity
    
def mask_Candidates(DCO_type, galaxy_mass_solar_masses, virial_radius_kpc, metallicity):
    
    if DCO_type == "BHNS":

        Escape_Velocity = calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc)
        maskCandidates_BHNS = (((Delay_time_BHNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS * Delay_time_BHNS * 3.2 * (10**13) < 0.1 * virial_radius_kpc * 3.08 * (10**16))) | ((Delay_time_BHNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS < Escape_Velocity))) & (metallicity <= (0.0142/5))
        seeds_candidates_BHNS = DCOseeds_BHNS[maskCandidates_BHNS]
        mask_seeds_candidates_BHNS = np.in1d(DCOseeds_BHNS, seeds_candidates_BHNS)
        
        percentage = round(((len(seeds_candidates_BHNS) / len(mask_seeds_candidates_BHNS))*100), 2)
        
        print("The percentage of candidate BHNS is: " + str(percentage) + "%")
        
        return mask_seeds_candidates_BHNS
        
    elif DCO_type == "BNS":
        
        Escape_Velocity = calculate_escape_velocity(galaxy_mass_solar_masses, virial_radius_kpc)
        maskCandidates_BNS = (((Delay_time_BNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BNS * Delay_time_BNS* 3.2 * (10**13) < 0.1 * virial_radius_kpc * 3.08 * (10**16))) | ((Delay_time_BNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BNS < Escape_Velocity))) & (metallicity <= (0.0142/5))
        seeds_candidates_BNS = fdata_BNS['doubleCompactObjects']['seed'][...].squeeze()[maskCandidates_BNS]
        mask_seeds_candidates_BNS = np.in1d(DCOseeds_BNS, seeds_candidates_BNS)
        
        percentage = round(((len(seeds_candidates_BNS) / len(mask_seeds_candidates_BNS))*100), 2)
        
        print("The percentage of candidate BNS is: " + str(percentage) + "%")
        
        return mask_seeds_candidates_BNS
    
mask_candidate_BHNS = mask_Candidates(DCO_type = "BHNS", galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6, metallicity=Data_BHNS.metallicitySystems)

mask_candidate_BHNS_2 = mask_Candidates(DCO_type = "BHNS", galaxy_mass_solar_masses = (10**8),  virial_radius_kpc = 1.3, metallicity=Data_BHNS.metallicitySystems)

mask_candidate_BNS = mask_Candidates(DCO_type = "BNS", galaxy_mass_solar_masses = (10**9),  virial_radius_kpc = 4.6, metallicity=Data_BNS.metallicitySystems)

mask_candidate_BNS_2 = mask_Candidates(DCO_type = "BNS", galaxy_mass_solar_masses = (10**8),  virial_radius_kpc = 1.3, metallicity=Data_BNS.metallicitySystems)

#### JUST BHNS:

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

Max_Reticulum_II_Mass = 2.6*(10**4)
Min_Reticulum_II_Mass = 10.4*(10**4)

Max_Tucana_III_Mass = 4.6*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

plt.figure(figsize=(10,6))
plt.grid()
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='green', alpha=0.75, lw=0, label = "Reticulum II Progenitor Stellar Mass")
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.75, lw=0, label = "Tucana III Progenitor Stellar Mass")
plt.scatter(np.asarray(metallicity),rates, s=16, marker = '.', label = "Candidate BHNS")
plt.scatter(np.asarray(metallicity),(rates*(10**0.7)), s=16, c='navy', marker = '.')
plt.yscale('log')
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.legend(markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

################################################

#### JUST BNS:

"""metallicitySystems_BNS = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS])
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

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

plt.figure(figsize=(10,6))
plt.grid()
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='green', alpha=0.75, lw=0, label = "Reticulum II Progenitor Stellar Mass")
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.75, lw=0, label = "Tucana III Progenitor Stellar Mass")
plt.scatter(np.asarray(metallicity_BNS),rates_BNS, s=16, c='red', marker = '.', label = "Candidate BNS")
plt.scatter(np.asarray(metallicity_BNS),(rates_BNS*(10**0.7)), s=16, c='darkred', marker = '.')
plt.yscale('log')
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

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

plt.figure(figsize=(10,6))
plt.grid()
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='green', alpha=0.75, lw=0, label = "Reticulum II Progenitor Stellar Mass")
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.75, lw=0, label = "Tucana III Progenitor Stellar Mass")
plt.scatter(np.asarray(metallicity),rates, s=16, marker = '.', label = "Candidate BHNS")
plt.scatter(np.asarray(metallicity),(rates*(10**0.7)), s=16, c='navy', marker = '.')
plt.scatter(np.asarray(metallicity_BNS),rates_BNS, s=16, c='red', marker = '.', label = "Candidate BNS")
plt.scatter(np.asarray(metallicity_BNS),(rates_BNS*(10**0.7)), s=16, c='darkred', marker = '.')
plt.yscale('log')
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.legend(markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""


## TO MAKE SEMI-MAJOR AXIS VS ECCENTRICITY PLOT WITH CANDIDATES MARKED:

"""candidate_semi_major_axis = Semi_major_axis_BHNS[mask_candidate_BHNS]
candidate_eccentricities = Eccentricities_BHNS[mask_candidate_BHNS]
metallicitySystems = Data_BHNS.metallicitySystems[mask_candidate_BHNS]

plt.scatter(Semi_major_axis_BHNS, Eccentricities_BHNS, s=4, c='gray', marker = '.')
plt.scatter(candidate_semi_major_axis, candidate_eccentricities, s=4, c=metallicitySystems,  marker = '.')
plt.xscale('log')
plt.xlabel(r'(a)$[R_{\odot}]$')
plt.ylabel('Eccentricity')
plt.colorbar(label='metallicity')
plt.legend(["ALL BHNS" , "Candidate BHNS"], markerscale = 4, ncol = 2 , loc = "lower left")
plt.show()"""

###TESTING: 

metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
print(Data_BHNS.metallicitySystems)

metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)

######### 

metallicitySystems_BHNS_2 = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS_2])
WEIGHTS_BHNS_2 = weights_BHNS[mask_candidate_BHNS_2]
print(Data_BHNS.metallicitySystems)

metallicity_2, frequency_2 = np.unique(metallicitySystems_BHNS_2, return_counts=True)

weights_sum_BHNS_2 = []

for X in np.asarray(metallicity_2):
    mask_metallicitySystems_BHNS_2 = np.in1d(metallicitySystems_BHNS_2, np.asarray(X))
    WEIGHTS_BHNS_masked_2 = WEIGHTS_BHNS_2[mask_metallicitySystems_BHNS_2] 
    weights_sum_BHNS_2 = np.append(weights_sum_BHNS_2, np.sum(WEIGHTS_BHNS_masked_2))

rates_2 = np.divide(weights_sum_BHNS_2, 77598320)

######### 

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

######### 

metallicitySystems_BNS_2 = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS_2])
WEIGHTS_BNS_2 = weights_BNS[mask_candidate_BNS_2]
print(Data_BNS.metallicitySystems)

metallicity_BNS_2, frequency_BNS_2 = np.unique(metallicitySystems_BNS_2, return_counts=True)

weights_sum_BNS_2 = []

for x_2 in np.asarray(metallicity_BNS_2):
    mask_metallicitySystems_BNS_2 = np.in1d(metallicitySystems_BNS_2, np.asarray(x_2))
    WEIGHTS_BNS_masked_2 = WEIGHTS_BNS_2[mask_metallicitySystems_BNS_2] 
    weights_sum_BNS_2 = np.append(weights_sum_BNS_2, np.sum(WEIGHTS_BNS_masked_2))

rates_BNS_2 = np.divide(weights_sum_BNS_2, 77598320)

######### 

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

y_error = [np.subtract(rates, rates_2), rates*0]
y_error_BNS = [np.subtract(rates_BNS, rates_BNS_2), rates_BNS*0]

print(np.unique(Data_BNS.metallicitySystems))
print(rates_BNS)

plt.figure(figsize=(15,9))
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.6, lw=0)

plt.errorbar(x=np.asarray(metallicity_BNS),y=rates_BNS, yerr=y_error_BNS, marker = '.', label = "Candidate BHNS", fmt='none', elinewidth=6, c='gold', alpha=0.7)
plt.errorbar(x=np.asarray(metallicity_BNS),y=rates_BNS*(10**0.7), yerr=y_error_BNS, marker = '.', fmt='none', elinewidth=6, c='darkgoldenrod', alpha=0.7)

plt.errorbar(x=np.asarray(metallicity),y=rates, yerr=y_error, marker = '.', label = "Candidate BHNS", fmt='none', elinewidth=6, c='teal', alpha=0.7)
plt.errorbar(x=np.asarray(metallicity),y=rates*(10**0.7), yerr=y_error, marker = '.',  fmt='none', elinewidth=6, c='teal')

plt.axvline(x=np.log10(0.0142), color='black', c='grey', linestyle=':')
plt.axvline(x=np.log10(0.0142/10), color='black',  c='grey', linestyle=':')

plt.text(np.log10(0.0146), 0.00000000050, r'$\rm{Z} \ = \ \rm{Z}_{\odot} $' , fontsize=10, rotation=90, verticalalignment='bottom')
plt.text(np.log10(0.0146/10), 0.00000000050, r'$\rm{Z} \ = \ \rm{Z}_{\odot}/10 $' , fontsize=10, rotation=90, verticalalignment='bottom')

plt.text(-4.1, 1.12/Min_Tucana_III_Mass, 'Tucana III Required Rates' , fontsize=10)
plt.text(-4.1, 1.05/Max_Reticulum_II_Mass, 'Reticulum II Required Rates' , fontsize=10)

plt.arrow(np.log10(0.03), rates_BNS[-1], 0, 9.04e-7 - rates_BNS[-1] - 0.0000001, head_length=0.0000001, width=0.01, linewidth=0, color='red', alpha=0.7)
plt.text(np.log10(0.033), 1.54e-07, r'20% of UFDs are' + "\n" + 'r-process enriched' , fontsize=7, rotation=90, verticalalignment='bottom', color='red', weight='bold', horizontalalignment='center')

x_pos = [np.asarray(metallicity)[0], np.asarray(metallicity)[1]]
x_pos1 = np.asarray(x_pos)
max1 = rates.max()*(10**0.7)
max2 = rates_BNS.max()*(10**0.7)
min1=rates_2.min()
min2=rates_BNS_2.min()
y_pos = [max1, max2]
y_pos1 = np.asarray(y_pos)
err = [max1 - min1, max2 - min2]
yerr = [np.asarray(err), np.asarray(err)*0]

plt.errorbar(x=x_pos1,y=y_pos1, yerr=np.asarray(yerr), marker = '.',  fmt='none', elinewidth=6, c='teal')

plt.yscale('log')
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.grid()
plt.legend()
plt.show()

################# MORE TESTING 

"""metallicitySystems_BHNS = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS])
WEIGHTS_BHNS = weights_BHNS[mask_candidate_BHNS]
print(Data_BHNS.metallicitySystems)
print(len(WEIGHTS_BHNS))
metallicity, frequency = np.unique(metallicitySystems_BHNS, return_counts=True)

weights_sum_BHNS = []

for n in np.asarray(metallicity):
    mask_metallicitySystems_BHNS = np.in1d(metallicitySystems_BHNS, np.asarray(n))
    WEIGHTS_BHNS_masked = WEIGHTS_BHNS[mask_metallicitySystems_BHNS] 
    weights_sum_BHNS = np.append(weights_sum_BHNS, np.sum(WEIGHTS_BHNS_masked))

rates = np.divide(weights_sum_BHNS, 77598320)
print(rates)
######### 

metallicitySystems_BHNS_2 = np.log10(Data_BHNS.metallicitySystems[mask_candidate_BHNS_2])
WEIGHTS_BHNS_2 = weights_BHNS[mask_candidate_BHNS_2]
print(Data_BHNS.metallicitySystems)

metallicity_2, frequency_2 = np.unique(metallicitySystems_BHNS_2, return_counts=True)

weights_sum_BHNS_2 = []

for X in np.asarray(metallicity_2):
    mask_metallicitySystems_BHNS_2 = np.in1d(metallicitySystems_BHNS_2, np.asarray(X))
    WEIGHTS_BHNS_masked_2 = WEIGHTS_BHNS_2[mask_metallicitySystems_BHNS_2] 
    weights_sum_BHNS_2 = np.append(weights_sum_BHNS_2, np.sum(WEIGHTS_BHNS_masked_2))

rates_2 = np.divide(weights_sum_BHNS_2, 77598320)

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

y_error = [np.subtract(rates, rates_2), rates*0]

ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10,10), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.6, lw=0)

axe.errorbar(x=np.asarray(metallicity),y=rates, yerr=y_error, marker = '.', label = "Candidate BHNS", fmt='none', elinewidth=6, c='teal', alpha=0.7)
axe.errorbar(x=np.asarray(metallicity),y=rates*(10**0.7), yerr=y_error, marker = '.',  fmt='none', elinewidth=6, c='teal')

axe.axvline(x=np.log10(0.0142), color='black', c='grey', linestyle=':')
axe.axvline(x=np.log10(0.0142/10), color='black',  c='grey', linestyle=':')

axe.text(np.log10(0.0146), 0.00000000050, r'$\rm{Z} \ = \ \rm{Z}_{\odot} $' , fontsize=10, rotation=90, verticalalignment='bottom')
axe.text(np.log10(0.0146/10), 0.00000000050, r'$\rm{Z} \ = \ \rm{Z}_{\odot}/10 $' , fontsize=10, rotation=90, verticalalignment='bottom')

axe.text(-4.1, 1.12/Min_Tucana_III_Mass, 'Tucana III Required Rates' , fontsize=9)
axe.text(-4.1, 1.05/Max_Reticulum_II_Mass, 'Reticulum II Required Rates' , fontsize=9)

axe.arrow(np.log10(0.0001), rates[0], 0, 7.725e-7 - rates[0] - 0.0000001, head_length=0.0000001, width=0.01, linewidth=0, color='red', alpha=0.7)
axe.text(np.log10(0.00009), 1.13e-07, r'20% of UFDs are' + "\n" + 'r-process enriched' , fontsize=7, rotation=90, verticalalignment='bottom', color='red', weight='bold', horizontalalignment='center')

axe.set_yscale('log')
layoutAxes(ax=axe, nameX=nameX, nameY=nameY)
axe.legend()
axe.grid()
plt.show()"""

################# EVEN MORE TESTING

"""metallicitySystems_BNS = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS])
WEIGHTS_BNS = weights_BNS[mask_candidate_BNS]
print(Data_BNS.metallicitySystems)

metallicity_BNS, frequency_BNS = np.unique(metallicitySystems_BNS, return_counts=True)

weights_sum_BNS = []

for x in np.asarray(metallicity_BNS):
    mask_metallicitySystems_BNS = np.in1d(metallicitySystems_BNS, np.asarray(x))
    WEIGHTS_BNS_masked = WEIGHTS_BNS[mask_metallicitySystems_BNS] 
    weights_sum_BNS = np.append(weights_sum_BNS, np.sum(WEIGHTS_BNS_masked))

rates_BNS = np.divide(weights_sum_BNS, 77598320)

######### 

metallicitySystems_BNS_2 = np.log10(Data_BNS.metallicitySystems[mask_candidate_BNS_2])
WEIGHTS_BNS_2 = weights_BNS[mask_candidate_BNS_2]
print(Data_BNS.metallicitySystems)

metallicity_BNS_2, frequency_BNS_2 = np.unique(metallicitySystems_BNS_2, return_counts=True)

weights_sum_BNS_2 = []

for x_2 in np.asarray(metallicity_BNS_2):
    mask_metallicitySystems_BNS_2 = np.in1d(metallicitySystems_BNS_2, np.asarray(x_2))
    WEIGHTS_BNS_masked_2 = WEIGHTS_BNS_2[mask_metallicitySystems_BNS_2] 
    weights_sum_BNS_2 = np.append(weights_sum_BNS_2, np.sum(WEIGHTS_BNS_masked_2))

rates_BNS_2 = np.divide(weights_sum_BNS_2, 77598320)

######### 

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

y_error_BNS = [np.subtract(rates_BNS, rates_BNS_2), rates_BNS*0]

plt.figure(figsize=(10,6))
plt.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
plt.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='grey', alpha=0.6, lw=0)

plt.errorbar(x=np.asarray(metallicity_BNS),y=rates_BNS, yerr=y_error_BNS, marker = '.', label = "Candidate BNS", fmt='none', elinewidth=6, c='gold', alpha=0.7)
plt.errorbar(x=np.asarray(metallicity_BNS),y=rates_BNS*(10**0.7), yerr=y_error_BNS, marker = '.', fmt='none', elinewidth=6, c='darkgoldenrod', alpha=0.7)

plt.axvline(x=np.log10(0.0142), color='black', c='grey', linestyle=':')
plt.axvline(x=np.log10(0.0142/10), color='black',  c='grey', linestyle=':')

plt.text(np.log10(0.0146), 1.8e-8, r'$\rm{Z} \ = \ \rm{Z}_{\odot} $' , fontsize=10, rotation=90, verticalalignment='bottom')
plt.text(np.log10(0.0146/10), 1.8e-8, r'$\rm{Z} \ = \ \rm{Z}_{\odot}/10 $' , fontsize=10, rotation=90, verticalalignment='bottom')

plt.text(-4.1, 1.12/Min_Tucana_III_Mass, 'Tucana III Required Rates' , fontsize=10)
plt.text(-4.1, 1.05/Max_Reticulum_II_Mass, 'Reticulum II Required Rates' , fontsize=10)

plt.arrow(np.log10(0.03), rates_BNS[-1], 0, 9.04e-7 - rates_BNS[-1] - 0.0000001, head_length=0.0000001, width=0.01, linewidth=0, color='red', alpha=0.7)
plt.text(np.log10(0.033), 1.86e-07, r'20% of UFDs are' + "\n" + 'r-process enriched' , fontsize=8, rotation=90, verticalalignment='bottom', color='red', weight='bold', horizontalalignment='center')

plt.yscale('log')
plt.xlabel(nameX)
plt.ylabel(nameY)
plt.grid()
plt.legend()
plt.show()"""

