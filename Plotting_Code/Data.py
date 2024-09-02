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

### To get PATHS: 

def obtainM1BHandM2BHassymetric(m1, m2):
    m1bh, m2bh = np.zeros_like(m1), np.zeros_like(m1)
    maskm1heavier = ( m1 >= m2)
    maskm2heavier = (m1 < m2)
    
    m1bh[maskm1heavier] = m1[maskm1heavier] 
    m1bh[maskm2heavier] = m2[maskm2heavier]
    m2bh[maskm1heavier] = m2[maskm1heavier]
    m2bh[maskm2heavier] = m1[maskm2heavier]
    
    return m1bh, m2bh # m1bh has all the heaviest systems

def mask_Candidates(UFDMass, Delay_time_BHNS, Systemic_Velocities_of_Interest_BHNS, DCOseeds_BHNS, metallicity):
            
        min_escape_velocity = math.sqrt(2*6.67*(10**-11)*(10**8)*1.989*(10**30)/(1.3 * 3.08 * (10**19)))/1000
        max_escape_velocity = math.sqrt(2*6.67*(10**-11)*(10**9)*1.989*(10**30)/(4.6 * 3.08 * (10**19)))/1000
        
        if UFDMass == 'min': 
            
            Escape_Velocity = min_escape_velocity
            maskCandidates_BHNS = (((Delay_time_BHNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS * Delay_time_BHNS * 3.2 * (10**13) < 0.1 * 1.3 * 3.08 * (10**16))) | ((Delay_time_BHNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS < Escape_Velocity))) & (metallicity <= (0.0142/5)) 
            seeds_candidates_BHNS = DCOseeds_BHNS[maskCandidates_BHNS]
            mask_seeds_candidates_BHNS_min = np.in1d(DCOseeds_BHNS, seeds_candidates_BHNS)
            
            return mask_seeds_candidates_BHNS_min
        
        if UFDMass == 'max': 
            
            Escape_Velocity = max_escape_velocity
            maskCandidates_BHNS = (((Delay_time_BHNS < (1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS * Delay_time_BHNS * 3.2 * (10**13) < 0.1 * 1.3 * 3.08 * (10**16))) | ((Delay_time_BHNS <(1*(10**3))) & (Systemic_Velocities_of_Interest_BHNS < Escape_Velocity))) & (metallicity <= (0.0142/5)) 
            seeds_candidates_BHNS = DCOseeds_BHNS[maskCandidates_BHNS]
            mask_seeds_candidates_BHNS_max = np.in1d(DCOseeds_BHNS, seeds_candidates_BHNS)
            
            return mask_seeds_candidates_BHNS_max

def Rates(metallicity_array, mask, weights_array, m_BH, m_NS): 
    
    r_NS = 13
    X_EFF = 0.5
    
    m_BH_useful = m_BH[mask]
    m_NS_useful = m_NS[mask]
    
    Mej_list = np.asarray([])
    for ind_chi, M_BH in enumerate(m_BH_useful):
        M_NS = m_NS_useful[ind_chi]
        Mej = calculateEjectedMassMerger(m_ns=M_NS, r_ns=r_NS, m_bh=M_BH, Xeff=X_EFF)
        Mej_list = np.append(Mej_list, np.asarray(Mej))
    
    mask_final = Mej_list > 0
    
    metallicities_before = metallicity_array[mask]
    metallicities = np.log10(metallicities_before[mask_final])
    
    weights_before = weights_array[mask]
    weights = weights_before[mask_final]
    metallicity, frequency = np.unique(metallicities, return_counts=True)
    
    print(len(Mej_list))
    print(len(metallicities))
    
    weights_sum = []
    
    for n in np.asarray(metallicity):
        mask_metallicity = np.in1d(metallicities, np.asarray(n))
        weights_masked = weights[mask_metallicity]
        weights_sum = np.append(weights_sum, np.sum(weights_masked))
    
    rates = np.divide(weights_sum, 77598320)
    
    return rates

def calculateRisco(m_bhtemp, Xefftemp):
    # this is prograde orbit
    # see also https://duetosymmetry.com/tool/kerr-isco-calculator/

    # everything in cgs
    c = 2.99792458E10 #[cm s-1] 
    G = 6.67259E-8   
    Msun = 1.99E33 # gr
    Rsun = 6.96E10 # cm     
    
    factorFront =   ((G*m_bhtemp)/c**2) #m_bhtemp #s
    
    Z1 = 1 + (1 - Xefftemp**2)**(1/3) * ((1 + Xefftemp)**(1/3) + (1 - Xefftemp)**(1/3) )
    Z2 = np.sqrt((3* Xefftemp**2 + Z1**2))
    
    Risco = factorFront * (3 + Z2 - np.sqrt((3-Z1)*(3+Z1 +2*Z2)))
    return Risco

def calculateEjectedMassMerger(m_ns, r_ns, m_bh, Xeff ):
 # from 1807.00011, Eq 4 
    # returns M_rem in solar masses 
    # input r and m in solar masses and R sun. Xeff in [0,1] (symmetric) 
    # RNS in km
    
    
    # everything in cgs
    c = 2.99792458E10 #[cm s-1] 
    G = 6.67259E-8   
    Msun = 1.99E33 # gr
    Rsun = 6.96E10 # cm         
    
    
    # convert to cgs
    r_ns  = r_ns*0.1*10**6 #np.asarray([1.E6]* len(m_ns)) # to cm
    m_ns_cgs = Msun * m_ns
    m_bh_cgs = Msun * m_bh
    
    
    alpha, beta, gamma, delta = 0.406, 0.139, 0.255, 1.761
    C_NS = G * m_ns_cgs / (r_ns * c**2)
    
    R_isco = calculateRisco(m_bh_cgs, Xeff)
    
    R_isco_norm  = R_isco / (m_bh_cgs * (G/c**2)) 
    
    Q = m_bh_cgs / m_ns_cgs
    
    eta = Q / (1 + Q)**2
    
    FirstTerm  = alpha*(1 - 2*C_NS) / eta**(1/3)
    SecondTerm = beta* R_isco_norm * C_NS / eta 
    
    A = np.asarray(FirstTerm - SecondTerm + gamma)
    B = np.zeros_like(m_ns_cgs)
    
    Mrem_model = np.maximum(A,B)**(delta)
    
    Mrem_model /= Msun # in solar masses 
    
    # and the true M remnant mass (not normalized and in solar masses =)
    Mrem_solar = Mrem_model * m_ns_cgs  
    return Mrem_solar # in [Msun]


class GetDATA (object):
    
    def __init__(self, models=[]):
        
        paths = np.asarray([])
        
        for x in models: 
            path = r'C:\Users\cauar\OneDrive\Área de Trabalho\Data\COMPASCompactOutput_BHNS_' + x + r'.h5'
            paths = np.append(paths, path)
            
            if x == 'A': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_A = x.mass1
                self.mass2_A = x. mass2
                self.weight_A = x. weight
                self.systemicVelocity_A = x. systemicVelocity
                self.tc_A = x. tc
                self.eccentricityDCOFormation_A = x. eccentricityDCOFormation
                self.separationDCOFormation_A = x. separationDCOFormation
                self.seeds_A = x. seeds
                self.metallicitySystems_A = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_A, m2=self.mass2_A)
                
                mask_A_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_A , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_A[1::2], DCOseeds_BHNS=self.seeds_A, metallicity=self.metallicitySystems_A)
                mask_A_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_A , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_A[1::2], DCOseeds_BHNS=self.seeds_A, metallicity=self.metallicitySystems_A)
                
                self.mask_A_max = mask_A_max
                self.mask_A_min = mask_A_min
                
                rates_A_max = Rates(metallicity_array=self.metallicitySystems_A, mask=mask_A_max, weights_array=self.weight_A, m_BH=m_bh, m_NS=m_ns)
                rates_A_min = Rates(metallicity_array=self.metallicitySystems_A, mask=mask_A_min, weights_array=self.weight_A, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_A_max = rates_A_max
                self.rates_A_min = rates_A_min
                
            if x == 'B': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_B = x.mass1
                self.mass2_B = x. mass2
                self.weight_B = x. weight
                self.systemicVelocity_B = x. systemicVelocity
                self.tc_B = x. tc
                self.eccentricityDCOFormation_B = x. eccentricityDCOFormation
                self.separationDCOFormation_B = x. separationDCOFormation
                self.seeds_B = x. seeds
                self.metallicitySystems_B = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_B, m2=self.mass2_B)
                
                mask_B_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_B , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_B[1::2], DCOseeds_BHNS=self.seeds_B, metallicity=self.metallicitySystems_B)
                mask_B_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_B , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_B[1::2], DCOseeds_BHNS=self.seeds_B, metallicity=self.metallicitySystems_B)
                
                self.mask_B_max = mask_B_max
                self.mask_B_min = mask_B_min
                
                rates_B_max = Rates(metallicity_array=self.metallicitySystems_B, mask=mask_B_max, weights_array=self.weight_B, m_BH=m_bh, m_NS=m_ns)
                rates_B_min = Rates(metallicity_array=self.metallicitySystems_B, mask=mask_B_min, weights_array=self.weight_B, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_B_max = rates_B_max
                self.rates_B_min = rates_B_min
            
            if x == 'C': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_C = x.mass1
                self.mass2_C = x. mass2
                self.weight_C = x. weight
                self.systemicVelocity_C = x. systemicVelocity
                self.tc_C = x. tc
                self.eccentricityDCOFormation_C = x. eccentricityDCOFormation
                self.separationDCOFormation_C = x. separationDCOFormation
                self.seeds_C = x. seeds
                self.metallicitySystems_C = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_C, m2=self.mass2_C)
                
                mask_C_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_C , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_C[1::2], DCOseeds_BHNS=self.seeds_C, metallicity=self.metallicitySystems_C)
                mask_C_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_C , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_C[1::2], DCOseeds_BHNS=self.seeds_C, metallicity=self.metallicitySystems_C)
                
                self.mask_C_max = mask_C_max
                self.mask_C_min = mask_C_min
                
                rates_C_max = Rates(metallicity_array=self.metallicitySystems_C, mask=mask_C_max, weights_array=self.weight_C, m_BH=m_bh, m_NS=m_ns)
                rates_C_min = Rates(metallicity_array=self.metallicitySystems_C, mask=mask_C_min, weights_array=self.weight_C, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_C_max = rates_C_max
                self.rates_C_min = rates_C_min
            
            if x == 'D': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_D = x.mass1
                self.mass2_D = x. mass2
                self.weight_D = x. weight
                self.systemicVelocity_D = x. systemicVelocity
                self.tc_D = x. tc
                self.eccentricityDCOFormation_D = x. eccentricityDCOFormation
                self.separationDCOFormation_D = x. separationDCOFormation
                self.seeds_D = x. seeds
                self.metallicitySystems_D = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_D, m2=self.mass2_D)
                
                mask_D_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_D , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_D[1::2], DCOseeds_BHNS=self.seeds_D, metallicity=self.metallicitySystems_D)
                mask_D_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_D , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_D[1::2], DCOseeds_BHNS=self.seeds_D, metallicity=self.metallicitySystems_D)
                
                self.mask_D_max = mask_D_max
                self.mask_D_min = mask_D_min
                
                rates_D_max = Rates(metallicity_array=self.metallicitySystems_D, mask=mask_D_max, weights_array=self.weight_D, m_BH=m_bh, m_NS=m_ns)
                rates_D_min = Rates(metallicity_array=self.metallicitySystems_D, mask=mask_D_min, weights_array=self.weight_D, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_D_max = rates_D_max
                self.rates_D_min = rates_D_min
            
            if x == 'F': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=True)
                x.setCOMPASData()
                
                self.mass1_F = x.mass1
                self.mass2_F = x. mass2
                self.weight_F = x. weight
                self.systemicVelocity_F = x. systemicVelocity
                self.tc_F = x. tc
                self.eccentricityDCOFormation_F = x. eccentricityDCOFormation
                self.separationDCOFormation_F = x. separationDCOFormation
                self.seeds_F = x. seeds
                self.metallicitySystems_F = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_F, m2=self.mass2_F)
                
                mask_F_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_F , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_F[1::2], DCOseeds_BHNS=self.seeds_F, metallicity=self.metallicitySystems_F)
                mask_F_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_F , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_F[1::2], DCOseeds_BHNS=self.seeds_F, metallicity=self.metallicitySystems_F)
                
                self.mask_F_max = mask_F_max
                self.mask_F_min = mask_F_min
                
                rates_F_max = Rates(metallicity_array=self.metallicitySystems_F, mask=mask_F_max, weights_array=self.weight_F, m_BH=m_bh, m_NS=m_ns)
                rates_F_min = Rates(metallicity_array=self.metallicitySystems_F, mask=mask_F_min, weights_array=self.weight_F, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_F_max = rates_F_max
                self.rates_F_min = rates_F_min
            
            if x == 'G': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_G = x.mass1
                self.mass2_G = x. mass2
                self.weight_G = x. weight
                self.systemicVelocity_G = x. systemicVelocity
                self.tc_G = x. tc
                self.eccentricityDCOFormation_G = x. eccentricityDCOFormation
                self.separationDCOFormation_G = x. separationDCOFormation
                self.seeds_G = x. seeds
                self.metallicitySystems_G = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_G, m2=self.mass2_G)
                
                mask_G_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_G , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_G[1::2], DCOseeds_BHNS=self.seeds_G, metallicity=self.metallicitySystems_G)
                mask_G_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_G , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_G[1::2], DCOseeds_BHNS=self.seeds_G, metallicity=self.metallicitySystems_G)
                
                self.mask_G_max = mask_G_max
                self.mask_G_min = mask_G_min
                
                rates_G_max = Rates(metallicity_array=self.metallicitySystems_G, mask=mask_G_max, weights_array=self.weight_G, m_BH=m_bh, m_NS=m_ns)
                rates_G_min = Rates(metallicity_array=self.metallicitySystems_G, mask=mask_G_min, weights_array=self.weight_G, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_G_max = rates_G_max
                self.rates_G_min = rates_G_min
            
            if x == 'H': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_H = x.mass1
                self.mass2_H = x. mass2
                self.weight_H = x. weight
                self.systemicVelocity_H = x. systemicVelocity
                self.tc_H = x. tc
                self.eccentricityDCOFormation_H = x. eccentricityDCOFormation
                self.separationDCOFormation_H = x. separationDCOFormation
                self.seeds_H = x. seeds
                self.metallicitySystems_H = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_H, m2=self.mass2_H)
                
                mask_H_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_H , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_H[1::2], DCOseeds_BHNS=self.seeds_H, metallicity=self.metallicitySystems_H)
                mask_H_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_H , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_H[1::2], DCOseeds_BHNS=self.seeds_H, metallicity=self.metallicitySystems_H)
                
                self.mask_H_max = mask_H_max
                self.mask_H_min = mask_H_min
                
                rates_H_max = Rates(metallicity_array=self.metallicitySystems_H, mask=mask_H_max, weights_array=self.weight_H, m_BH=m_bh, m_NS=m_ns)
                rates_H_min = Rates(metallicity_array=self.metallicitySystems_H, mask=mask_H_min, weights_array=self.weight_H, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_H_max = rates_H_max
                self.rates_H_min = rates_H_min
            
            if x == 'I': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_I = x.mass1
                self.mass2_I = x. mass2
                self.weight_I = x. weight
                self.systemicVelocity_I = x. systemicVelocity
                self.tc_I = x. tc
                self.eccentricityDCOFormation_I = x. eccentricityDCOFormation
                self.separationDCOFormation_I = x. separationDCOFormation
                self.seeds_I = x. seeds
                self.metallicitySystems_I = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_I, m2=self.mass2_I)
                
                mask_I_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_I , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_I[1::2], DCOseeds_BHNS=self.seeds_I, metallicity=self.metallicitySystems_I)
                mask_I_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_I , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_I[1::2], DCOseeds_BHNS=self.seeds_I, metallicity=self.metallicitySystems_I)
                
                self.mask_I_max = mask_I_max
                self.mask_I_min = mask_I_min
                
                rates_I_max = Rates(metallicity_array=self.metallicitySystems_I, mask=mask_I_max, weights_array=self.weight_I, m_BH=m_bh, m_NS=m_ns)
                rates_I_min = Rates(metallicity_array=self.metallicitySystems_I, mask=mask_I_min, weights_array=self.weight_I, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_I_max = rates_I_max
                self.rates_I_min = rates_I_min
            
            if x == 'J': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_J = x.mass1
                self.mass2_J = x. mass2
                self.weight_J = x. weight
                self.systemicVelocity_J = x. systemicVelocity
                self.tc_J = x. tc
                self.eccentricityDCOFormation_J = x. eccentricityDCOFormation
                self.separationDCOFormation_J = x. separationDCOFormation
                self.seeds_J = x. seeds
                self.metallicitySystems_J = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_J, m2=self.mass2_J)
                
                mask_J_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_J , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_J[1::2], DCOseeds_BHNS=self.seeds_J, metallicity=self.metallicitySystems_J)
                mask_J_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_J , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_J[1::2], DCOseeds_BHNS=self.seeds_J, metallicity=self.metallicitySystems_J)
                
                self.mask_J_max = mask_J_max
                self.mask_J_min = mask_J_min
                
                rates_J_max = Rates(metallicity_array=self.metallicitySystems_J, mask=mask_J_max, weights_array=self.weight_J, m_BH=m_bh, m_NS=m_ns)
                rates_J_min = Rates(metallicity_array=self.metallicitySystems_J, mask=mask_J_min, weights_array=self.weight_J, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_J_max = rates_J_max
                self.rates_J_min = rates_J_min
            
            if x == 'K': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=True)
                x.setCOMPASData()
                
                self.mass1_K = x.mass1
                self.mass2_K = x. mass2
                self.weight_K = x. weight
                self.systemicVelocity_K = x. systemicVelocity
                self.tc_K = x. tc
                self.eccentricityDCOFormation_K = x. eccentricityDCOFormation
                self.separationDCOFormation_K = x. separationDCOFormation
                self.seeds_K = x. seeds
                self.metallicitySystems_K = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_K, m2=self.mass2_K)
                
                mask_K_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_K , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_K[1::2], DCOseeds_BHNS=self.seeds_K, metallicity=self.metallicitySystems_K)
                mask_K_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_K , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_K[1::2], DCOseeds_BHNS=self.seeds_K, metallicity=self.metallicitySystems_K)
                
                self.mask_K_max = mask_K_max
                self.mask_K_min = mask_K_min
                
                rates_K_max = Rates(metallicity_array=self.metallicitySystems_K, mask=mask_K_max, weights_array=self.weight_K, m_BH=m_bh, m_NS=m_ns)
                rates_K_min = Rates(metallicity_array=self.metallicitySystems_K, mask=mask_K_min, weights_array=self.weight_K, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_K_max = rates_K_max
                self.rates_K_min = rates_K_min
            
            if x == 'L': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_L = x.mass1
                self.mass2_L = x. mass2
                self.weight_L = x. weight
                self.systemicVelocity_L = x. systemicVelocity
                self.tc_L = x. tc
                self.eccentricityDCOFormation_L = x. eccentricityDCOFormation
                self.separationDCOFormation_L = x. separationDCOFormation
                self.seeds_L = x. seeds
                self.metallicitySystems_L = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_L, m2=self.mass2_L)
                
                mask_L_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_L , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_L[1::2], DCOseeds_BHNS=self.seeds_L, metallicity=self.metallicitySystems_L)
                mask_L_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_L , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_L[1::2], DCOseeds_BHNS=self.seeds_L, metallicity=self.metallicitySystems_L)
                
                self.mask_L_max = mask_L_max
                self.mask_L_min = mask_L_min
                
                rates_L_max = Rates(metallicity_array=self.metallicitySystems_L, mask=mask_L_max, weights_array=self.weight_L, m_BH=m_bh, m_NS=m_ns)
                rates_L_min = Rates(metallicity_array=self.metallicitySystems_L, mask=mask_L_min, weights_array=self.weight_L, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_L_max = rates_L_max
                self.rates_L_min = rates_L_min
            
            if x == 'M': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_M = x.mass1
                self.mass2_M = x. mass2
                self.weight_M = x. weight
                self.systemicVelocity_M = x. systemicVelocity
                self.tc_M = x. tc
                self.eccentricityDCOFormation_M = x. eccentricityDCOFormation
                self.separationDCOFormation_M = x. separationDCOFormation
                self.seeds_M = x. seeds
                self.metallicitySystems_M = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_M, m2=self.mass2_M)
                
                mask_M_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_M , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_M[1::2], DCOseeds_BHNS=self.seeds_M, metallicity=self.metallicitySystems_M)
                mask_M_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_M , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_M[1::2], DCOseeds_BHNS=self.seeds_M, metallicity=self.metallicitySystems_M)
                
                self.mask_M_max = mask_M_max
                self.mask_M_min = mask_M_min
                
                rates_M_max = Rates(metallicity_array=self.metallicitySystems_M, mask=mask_M_max, weights_array=self.weight_M, m_BH=m_bh, m_NS=m_ns)
                rates_M_min = Rates(metallicity_array=self.metallicitySystems_M, mask=mask_M_min, weights_array=self.weight_M, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_M_max = rates_M_max
                self.rates_M_min = rates_M_min
            
            if x == 'N': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_N = x.mass1
                self.mass2_N = x. mass2
                self.weight_N = x. weight
                self.systemicVelocity_N = x. systemicVelocity
                self.tc_N = x. tc
                self.eccentricityDCOFormation_N = x. eccentricityDCOFormation
                self.separationDCOFormation_N = x. separationDCOFormation
                self.seeds_N = x. seeds
                self.metallicitySystems_N = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_N, m2=self.mass2_N)
                
                mask_N_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_N , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_N[1::2], DCOseeds_BHNS=self.seeds_N, metallicity=self.metallicitySystems_N)
                mask_N_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_N , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_N[1::2], DCOseeds_BHNS=self.seeds_N, metallicity=self.metallicitySystems_N)
                
                self.mask_N_max = mask_N_max
                self.mask_N_min = mask_N_min
                
                rates_N_max = Rates(metallicity_array=self.metallicitySystems_N, mask=mask_N_max, weights_array=self.weight_N, m_BH=m_bh, m_NS=m_ns)
                rates_N_min = Rates(metallicity_array=self.metallicitySystems_N, mask=mask_N_min, weights_array=self.weight_N, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_N_max = rates_N_max
                self.rates_N_min = rates_N_min
            
            if x == 'O': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_O = x.mass1
                self.mass2_O = x. mass2
                self.weight_O = x. weight
                self.systemicVelocity_O = x. systemicVelocity
                self.tc_O = x. tc
                self.eccentricityDCOFormation_O = x. eccentricityDCOFormation
                self.separationDCOFormation_O = x. separationDCOFormation
                self.seeds_O = x. seeds
                self.metallicitySystems_O = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_O, m2=self.mass2_O)
                
                mask_O_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_O , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_O[1::2], DCOseeds_BHNS=self.seeds_O, metallicity=self.metallicitySystems_O)
                mask_O_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_O , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_O[1::2], DCOseeds_BHNS=self.seeds_O, metallicity=self.metallicitySystems_O)
                
                self.mask_O_max = mask_O_max
                self.mask_O_min = mask_O_min
                
                rates_O_max = Rates(metallicity_array=self.metallicitySystems_O, mask=mask_O_max, weights_array=self.weight_O, m_BH=m_bh, m_NS=m_ns)
                rates_O_min = Rates(metallicity_array=self.metallicitySystems_O, mask=mask_O_min, weights_array=self.weight_O, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_O_max = rates_O_max
                self.rates_O_min = rates_O_min
            
            if x == 'P': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_P = x.mass1
                self.mass2_P = x. mass2
                self.weight_P = x. weight
                self.systemicVelocity_P = x. systemicVelocity
                self.tc_P = x. tc
                self.eccentricityDCOFormation_P = x. eccentricityDCOFormation
                self.separationDCOFormation_P = x. separationDCOFormation
                self.seeds_P = x. seeds
                self.metallicitySystems_P = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_P, m2=self.mass2_P)
                
                mask_P_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_P , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_P[1::2], DCOseeds_BHNS=self.seeds_P, metallicity=self.metallicitySystems_P)
                mask_P_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_P , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_P[1::2], DCOseeds_BHNS=self.seeds_P, metallicity=self.metallicitySystems_P)
                
                self.mask_P_max = mask_P_max
                self.mask_P_min = mask_P_min
                
                rates_P_max = Rates(metallicity_array=self.metallicitySystems_P, mask=mask_P_max, weights_array=self.weight_P, m_BH=m_bh, m_NS=m_ns)
                rates_P_min = Rates(metallicity_array=self.metallicitySystems_P, mask=mask_P_min, weights_array=self.weight_P, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_P_max = rates_P_max
                self.rates_P_min = rates_P_min
            
            if x == 'Q': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_Q = x.mass1
                self.mass2_Q = x. mass2
                self.weight_Q = x. weight
                self.systemicVelocity_Q = x. systemicVelocity
                self.tc_Q = x. tc
                self.eccentricityDCOFormation_Q = x. eccentricityDCOFormation
                self.separationDCOFormation_Q = x. separationDCOFormation
                self.seeds_Q = x. seeds
                self.metallicitySystems_Q = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_Q, m2=self.mass2_Q)
                
                mask_Q_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_Q , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_Q[1::2], DCOseeds_BHNS=self.seeds_Q, metallicity=self.metallicitySystems_Q)
                mask_Q_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_Q , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_Q[1::2], DCOseeds_BHNS=self.seeds_Q, metallicity=self.metallicitySystems_Q)
                
                self.mask_Q_max = mask_Q_max
                self.mask_Q_min = mask_Q_min
                
                rates_Q_max = Rates(metallicity_array=self.metallicitySystems_Q, mask=mask_Q_max, weights_array=self.weight_Q, m_BH=m_bh, m_NS=m_ns)
                rates_Q_min = Rates(metallicity_array=self.metallicitySystems_Q, mask=mask_Q_min, weights_array=self.weight_Q, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_Q_max = rates_Q_max
                self.rates_Q_min = rates_Q_min
            
            if x == 'R': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_R = x.mass1
                self.mass2_R = x. mass2
                self.weight_R = x. weight
                self.systemicVelocity_R = x. systemicVelocity
                self.tc_R = x. tc
                self.eccentricityDCOFormation_R = x. eccentricityDCOFormation
                self.separationDCOFormation_R = x. separationDCOFormation
                self.seeds_R = x. seeds
                self.metallicitySystems_R = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_R, m2=self.mass2_R)
                
                mask_R_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_R , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_R[1::2], DCOseeds_BHNS=self.seeds_R, metallicity=self.metallicitySystems_R)
                mask_R_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_R , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_R[1::2], DCOseeds_BHNS=self.seeds_R, metallicity=self.metallicitySystems_R)
                
                self.mask_R_max = mask_R_max
                self.mask_R_min = mask_R_min
                
                rates_R_max = Rates(metallicity_array=self.metallicitySystems_R, mask=mask_R_max, weights_array=self.weight_R, m_BH=m_bh, m_NS=m_ns)
                rates_R_min = Rates(metallicity_array=self.metallicitySystems_R, mask=mask_R_min, weights_array=self.weight_R, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_R_max = rates_R_max
                self.rates_R_min = rates_R_min
            
            if x == 'S': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_S = x.mass1
                self.mass2_S = x. mass2
                self.weight_S = x. weight
                self.systemicVelocity_S = x. systemicVelocity
                self.tc_S = x. tc
                self.eccentricityDCOFormation_S = x. eccentricityDCOFormation
                self.separationDCOFormation_S = x. separationDCOFormation
                self.seeds_S = x. seeds
                self.metallicitySystems_S = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_S, m2=self.mass2_S)
                
                mask_S_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_S , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_S[1::2], DCOseeds_BHNS=self.seeds_S, metallicity=self.metallicitySystems_S)
                mask_S_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_S , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_S[1::2], DCOseeds_BHNS=self.seeds_S, metallicity=self.metallicitySystems_S)
                
                self.mask_S_max = mask_S_max
                self.mask_S_min = mask_S_min
                
                rates_S_max = Rates(metallicity_array=self.metallicitySystems_S, mask=mask_S_max, weights_array=self.weight_S, m_BH=m_bh, m_NS=m_ns)
                rates_S_min = Rates(metallicity_array=self.metallicitySystems_S, mask=mask_S_min, weights_array=self.weight_S, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_S_max = rates_S_max
                self.rates_S_min = rates_S_min
            
            if x == 'T': 
                
                x=COMPASData(path=path, lazyData=True, Mlower=5., \
                 Mupper=150., binaryFraction=1)
                x.setCOMPASDCOmask(types='BHNS',  withinHubbleTime=True, optimistic=False)
                x.setCOMPASData()
                
                self.mass1_T = x.mass1
                self.mass2_T = x. mass2
                self.weight_T = x. weight
                self.systemicVelocity_T = x. systemicVelocity
                self.tc_T = x. tc
                self.eccentricityDCOFormation_T = x. eccentricityDCOFormation
                self.separationDCOFormation_T = x. separationDCOFormation
                self.seeds_T = x. seeds
                self.metallicitySystems_T = x.metallicitySystems
                
                m_bh, m_ns = obtainM1BHandM2BHassymetric(m1=self.mass1_T, m2=self.mass2_T)
                
                mask_T_max = mask_Candidates(UFDMass='max', Delay_time_BHNS=self.tc_T , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_T[1::2], DCOseeds_BHNS=self.seeds_T, metallicity=self.metallicitySystems_T)
                mask_T_min = mask_Candidates(UFDMass='min', Delay_time_BHNS=self.tc_T , Systemic_Velocities_of_Interest_BHNS=self.systemicVelocity_T[1::2], DCOseeds_BHNS=self.seeds_T, metallicity=self.metallicitySystems_T)
                
                self.mask_T_max = mask_T_max
                self.mask_T_min = mask_T_min
                
                rates_T_max = Rates(metallicity_array=self.metallicitySystems_T, mask=mask_T_max, weights_array=self.weight_T, m_BH=m_bh, m_NS=m_ns)
                rates_T_min = Rates(metallicity_array=self.metallicitySystems_T, mask=mask_T_min, weights_array=self.weight_T, m_BH=m_bh, m_NS=m_ns)
                
                self.rates_T_max = rates_T_max
                self.rates_T_min = rates_T_min
                    
        self. paths = paths


list_of_models = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

data = GetDATA(models=list_of_models)

try: 
    A_max_rate = np.max(data.rates_A_max)*(10**0.7)
    A_min_rate = np.min(data.rates_A_min)
    
    B_max_rate = np.max(data.rates_B_max)*(10**0.7)
    B_min_rate = np.min(data.rates_B_min)
    
    C_max_rate = np.max(data.rates_C_max)*(10**0.7)
    C_min_rate = np.min(data.rates_C_min)

    D_max_rate = np.max(data.rates_D_max)*(10**0.7)
    D_min_rate = np.min(data.rates_D_min)

    F_max_rate = np.max(data.rates_F_max)*(10**0.7)
    F_min_rate = np.min(data.rates_F_min)

    G_max_rate = np.max(data.rates_G_max)*(10**0.7)
    G_min_rate = np.min(data.rates_G_min)

    H_max_rate = np.max(data.rates_H_max)*(10**0.7)
    H_min_rate = np.min(data.rates_H_min)

    I_max_rate = np.max(data.rates_I_max)*(10**0.7)
    I_min_rate = np.min(data.rates_I_min)

    J_max_rate = np.max(data.rates_J_max)*(10**0.7)
    J_min_rate = np.min(data.rates_J_min)

    K_max_rate = np.max(data.rates_K_max)*(10**0.7)
    K_min_rate = np.min(data.rates_K_min)

    L_max_rate = np.max(data.rates_L_max)*(10**0.7)
    L_min_rate = np.min(data.rates_L_min)

    M_max_rate = np.max(data.rates_M_max)*(10**0.7)
    M_min_rate = np.min(data.rates_M_min)

    N_max_rate = np.max(data.rates_N_max)*(10**0.7)
    N_min_rate = np.min(data.rates_N_min)

    O_max_rate = np.max(data.rates_O_max)*(10**0.7)
    O_min_rate = np.min(data.rates_O_min)

    P_max_rate = np.max(data.rates_P_max)*(10**0.7)
    P_min_rate = np.min(data.rates_P_min)

    Q_max_rate = np.max(data.rates_Q_max)*(10**0.7)
    Q_min_rate = np.min(data.rates_Q_min)

    R_max_rate = np.max(data.rates_R_max)*(10**0.7)
    R_min_rate = np.min(data.rates_R_min)

    S_max_rate = np.max(data.rates_S_max)*(10**0.7)
    S_min_rate = np.min(data.rates_S_min)

    T_max_rate = np.max(data.rates_T_max)*(10**0.7)
    T_min_rate = np.min(data.rates_T_min)

except ValueError:  #raised if `y` is empty.
    A_max_rate = np.asarray([0])
    A_min_rate = np.asarray([0])
    
    B_max_rate = np.asarray([0])
    B_min_rate = np.asarray([0])
    
    C_max_rate = np.asarray([0])
    C_min_rate = np.asarray([0])
    
    D_max_rate = np.asarray([0])
    D_min_rate = np.asarray([0])
    
    F_max_rate = np.asarray([0])
    F_min_rate = np.asarray([0])
    
    G_max_rate = np.asarray([0])
    G_min_rate = np.asarray([0])
    
    H_max_rate = np.asarray([0])
    H_min_rate = np.asarray([0])
    
    I_max_rate = np.asarray([0])
    I_min_rate = np.asarray([0])
    
    J_max_rate = np.asarray([0])
    J_min_rate = np.asarray([0])
    
    K_max_rate = np.asarray([0])
    K_min_rate = np.asarray([0])
    
    L_max_rate = np.asarray([0])
    L_min_rate = np.asarray([0])
    
    M_max_rate = np.asarray([0])
    M_min_rate = np.asarray([0])
    
    N_max_rate = np.asarray([0])
    N_min_rate = np.asarray([0])
    
    O_max_rate = np.asarray([0])
    O_min_rate = np.asarray([0])
    
    P_max_rate = np.asarray([0])
    P_min_rate = np.asarray([0])
    
    Q_max_rate = np.asarray([0])
    Q_min_rate = np.asarray([0])
    
    R_max_rate = np.asarray([0])
    R_min_rate = np.asarray([0])
    
    S_max_rate = np.asarray([0])
    S_min_rate = np.asarray([0])
    
    T_max_rate = np.asarray([0])
    T_min_rate = np.asarray([0])
    
y_positions = np.asarray([A_max_rate, B_max_rate, C_max_rate, D_max_rate, F_max_rate, G_max_rate, H_max_rate, I_max_rate, J_max_rate, K_max_rate, L_max_rate, M_max_rate, N_max_rate, O_max_rate, P_max_rate, Q_max_rate, R_max_rate, S_max_rate, T_max_rate])

min_y_positions = np.array([A_min_rate, B_min_rate, C_min_rate, D_min_rate, F_min_rate, G_min_rate, H_min_rate, I_min_rate, J_min_rate, K_min_rate, L_min_rate, M_min_rate, N_min_rate, O_min_rate, P_min_rate, Q_min_rate, R_min_rate, S_min_rate, T_min_rate])

err = np.subtract(y_positions, min_y_positions)

yerror = [err, err*0]

rates_A = np.append(data.rates_A_max*(10**0.7), data.rates_A_min)
rates_B = np.append(data.rates_B_max*(10**0.7), data.rates_B_min)
rates_C = np.append(data.rates_C_max*(10**0.7), data.rates_C_min)
rates_D = np.append(data.rates_D_max*(10**0.7), data.rates_D_min)
rates_F = np.append(data.rates_F_max*(10**0.7), data.rates_F_min)
rates_G = np.append(data.rates_G_max*(10**0.7), data.rates_G_min)
rates_H = np.append(data.rates_H_max*(10**0.7), data.rates_H_min)
rates_I = np.append(data.rates_I_max*(10**0.7), data.rates_I_min)
rates_J = np.append(data.rates_J_max*(10**0.7), data.rates_J_min)
rates_K = np.append(data.rates_K_max*(10**0.7), data.rates_K_min)
rates_L = np.append(data.rates_L_max*(10**0.7), data.rates_L_min)
rates_M = np.append(data.rates_M_max*(10**0.7), data.rates_M_min)
rates_N = np.append(data.rates_N_max*(10**0.7), data.rates_N_min)
rates_O = np.append(data.rates_O_max*(10**0.7), data.rates_O_min)
rates_P = np.append(data.rates_P_max*(10**0.7), data.rates_P_min)
rates_Q = np.append(data.rates_Q_max*(10**0.7), data.rates_Q_min)
rates_R = np.append(data.rates_R_max*(10**0.7), data.rates_R_min)
rates_S = np.append(data.rates_S_max*(10**0.7), data.rates_S_min)
rates_T = np.append(data.rates_T_max*(10**0.7), data.rates_T_min)

Rates_data = [rates_A, rates_B, rates_C, rates_D, rates_F, rates_G, rates_H, rates_I, rates_J, rates_K, rates_L, rates_M, rates_N, rates_O, rates_P, rates_Q, rates_R, rates_S, rates_T]

print(len(Rates_data))