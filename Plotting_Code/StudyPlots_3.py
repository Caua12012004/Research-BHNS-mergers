#import sys
# sys.path.append('../Scripts')
from PostProcessingScripts import * 
# import pandas as pd 
# import string 
# just to make the cells appear wider:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

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
from Data_Colombo import *
import EOS_properties as ep
import m_ejecta_disk_lambda as medl

## DELAY TIME VS SYSTEMIC VELOCITY PLOT: 

def get_final_mask(m_BH, m_NS, mask):
    
    Mbh = m_BH[mask]
    Mns = m_NS[mask]
    
    eos = "DD2"
    ens = ep.NS_EOS_properties(eos)
    spin_bh=0.5
    i_tilt = 0
    
    Mej_list = np.asarray([])
    for ind_chi, M_BH in enumerate(Mbh):
        M_NS = Mns[ind_chi]
        lamb_ns = ens.fun_lambda_kappa(M_NS)
        m_dyn = medl.Mej_Foucart20_lambda(M_BH,M_NS,spin_bh,i_tilt,lamb_ns) #dynamical ejecta mass based on Kruger&Foucart20 eq.9
        m_out = medl.M_out_f18_lambda(M_BH,M_NS,spin_bh,lamb_ns) #mass expelled based on Foucart+18 eq.4
        
        f = 0.5
        if (m_dyn>f*m_out):
            m_dyn = f*m_out
        
        Mej = np.maximum(m_out - m_dyn,0.)

        Mej_list = np.append(Mej_list, np.asarray(Mej))
    
    final_mask = Mej_list > 0
    
    return final_mask


models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model F', 'Model G', 'Model H', 'Model I', 'Model J', 'Model K', 'Model L', 'Model M', 'Model N', 'Model O', 'Model P', 'Model Q', 'Model R', 'Model S', 'Model T']

"""ncols, nrows = 3,6
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 40), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharey=True, sharex=True, layout='constrained')
f.set_constrained_layout_pads(w_pad=2.2, wspace=0.0, hspace=0.0, h_pad=0.08) 

n= [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
x= [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
y=0

for ind, delay_time in enumerate(delay_times_data[1:], 1):
    
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities_data[ind]), y=np.log10(delay_time), c='grey', s=1)
    
    mask = mask_data[ind]
    final_mask = get_final_mask(m_BH=mass_bh_data[ind], m_NS=mass_ns_data[ind], mask=mask)
    
    systemic_velocities = systemic_velocities_data[ind]
    
    seeds_original = seeds_data[ind]
    
    mask_seeds_1 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[ind][0])
    mask_seeds_2 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[ind][1])
    mask_seeds_3 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[ind][2])
    mask_seeds_4 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[ind][3])
    mask_seeds_5 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[ind][4])
    
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask][mask_seeds_1]), y=np.log10(delay_time[mask][final_mask][mask_seeds_1]), c='blue', s=1)
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask][mask_seeds_2]), y=np.log10(delay_time[mask][final_mask][mask_seeds_2]), c='red', s=1)
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask][mask_seeds_3]), y=np.log10(delay_time[mask][final_mask][mask_seeds_3]), c='yellow', s=1)
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask][mask_seeds_4]), y=np.log10(delay_time[mask][final_mask][mask_seeds_4]), c='darkslategrey', s=1)
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask][mask_seeds_5]), y=np.log10(delay_time[mask][final_mask][mask_seeds_5]), c='black', s=1)
    
    axe[x[y], n[y]].text(0.05, 0.1, models[ind], horizontalalignment='left', verticalalignment='bottom', transform=axe[x[y], n[y]].transAxes)
    
    if not models[ind] in ['Model R', 'Model S', 'Model T']:
        axe[x[y], n[y]].tick_params(axis='x', which='both', bottom=False)
    
    if not models[ind] in ['Model B', 'Model F', 'Model I', 'Model L', 'Model O', 'Model R']:
        axe[x[y], n[y]].tick_params(axis='y', which='both', left=False)
    
    y=y+1
    
    model = list_of_models[ind]
    
    print(f'The number of systems in the classic channel for model {model} is: {len(systemic_velocities[mask][final_mask][mask_seeds_1])} - The percentage is {round(((len(systemic_velocities[mask][final_mask][mask_seeds_1])/len(seeds_original[mask][final_mask]))*100),1)}%')
    print(f'The number of systems in the only stable mass transfer channel for model {model} is: {len(systemic_velocities[mask][final_mask][mask_seeds_2])} - The percentage is {round(((len(systemic_velocities[mask][final_mask][mask_seeds_2])/len(seeds_original[mask][final_mask]))*100),1)}%')
    print(f'The number of systems in the double core CEE channel for model {model} is: {len(systemic_velocities[mask][final_mask][mask_seeds_3])} - The percentage is {round(((len(systemic_velocities[mask][final_mask][mask_seeds_3])/len(seeds_original[mask][final_mask]))*100),1)}%')
    print(f'The number of systems in the single core CEE channel for model {model} is: {len(systemic_velocities[mask][final_mask][mask_seeds_4])} - The percentage is {round(((len(systemic_velocities[mask][final_mask][mask_seeds_4])/len(seeds_original[mask][final_mask]))*100),1)}%')
    print(f'The number of systems in the other channels for model {model} is: {len(systemic_velocities[mask][final_mask][mask_seeds_5])} - The percentage is {round(((len(systemic_velocities[mask][final_mask][mask_seeds_5])/len(seeds_original[mask][final_mask]))*100),1)}%')
    print("  ")

nameX = r'$\log_{{\rm{10}}}(v)$$[km/s]$'
nameY = r'$\log_{{\rm{10}}}(t_{merge})$$[Myr]$'
    
f.text(0.5, 0.03, nameX, ha='center', size='large')
f.text(0.02, 0.5, nameY, va='center', rotation='vertical', size='large')

plt.show()"""

#### JUST A TEST: 

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

"""ncols, nrows = 3,6
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 40), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharey=True, sharex=True, layout='constrained')
f.set_constrained_layout_pads(w_pad=2.2, wspace=0.0, hspace=0.0, h_pad=0.08) 

n= [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
x= [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
y=0

def sum(metallicity_array, mask_1, mask_2, weights_array, model): 
    
    metallicities_before = metallicity_array[mask_1]
    metallicities = np.log10(metallicities_before[mask_2])
    
    weights_before = weights_array[mask_1]
    weights = weights_before[mask_2]
    metallicity, frequency = np.unique(metallicities, return_counts=True)
    
    #print(f'The number of Candidates in Model {model_letter} is: {len(metallicities)}')
    
    weights_sum = []
    
    for n in np.asarray(metallicity):
        mask_metallicity = np.in1d(metallicities, np.asarray(n))
        weights_masked = weights[mask_metallicity]
        weights_sum = np.append(weights_sum, np.sum(weights_masked))
    
    print(f'The total number of systems in model {model} is: {len(weights_sum)}')
    print('  ')
    
    return weights_sum

for ind, weights in enumerate(weights_data[1:], 1):
    
    mask = mask_data[ind]
    final_mask = get_final_mask(m_BH=mass_bh_data[ind], m_NS=mass_ns_data[ind], mask=mask)
    
    metallicities = metallicity_data[ind]
    met_final = np.log10(metallicities[mask][final_mask])
    
    weights = sum(metallicity_array=metallicities, mask_1=mask, mask_2=final_mask, weights_array=weights, model=list_of_models[ind])
    
    axe[x[y], n[y]].scatter(x=np.unique(met_final), y=weights, c='teal', s=1)
    
    axe[x[y], n[y]].text(0.05, 0.1, models[ind], horizontalalignment='left', verticalalignment='bottom', transform=axe[x[y], n[y]].transAxes)
    
    if not models[ind] in ['Model R', 'Model S', 'Model T']:
        axe[x[y], n[y]].tick_params(axis='x', which='both', bottom=False)
    
    if not models[ind] in ['Model B', 'Model F', 'Model I', 'Model L', 'Model O', 'Model R']:
        axe[x[y], n[y]].tick_params(axis='y', which='both', left=False)
    
    axe[x[y], n[y]].set_yscale('log')
    y=y+1

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'Sum of Weights'
    
f.text(0.5, 0.03, nameX, ha='center', size='large')
f.text(0.02, 0.5, nameY, va='center', rotation='vertical', size='large')

plt.show()"""

###Investigating model F:

"""ncols, nrows = 1,1
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 40), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

mask = mask_data[4]
final_mask = get_final_mask(m_BH=mass_bh_data[4], m_NS=mass_ns_data[4], mask=mask)

metallicities = metallicity_data[4]
weights = weights_data[4]

seeds_original = seeds_data[4]
    
mask_seeds_1 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][0])
mask_seeds_2 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][1])
mask_seeds_3 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][2])
mask_seeds_4 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][3])
mask_seeds_5 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][4])

plt.scatter(x=np.log10(metallicities[mask][final_mask][mask_seeds_1]), y=(weights[mask][final_mask][mask_seeds_1]), c='blue', s=9)
plt.scatter(x=np.log10(metallicities[mask][final_mask][mask_seeds_2]), y=(weights[mask][final_mask][mask_seeds_2]), c='red', s=9)
plt.scatter(x=np.log10(metallicities[mask][final_mask][mask_seeds_3]), y=(weights[mask][final_mask][mask_seeds_3]), c='yellow', s=9)
plt.scatter(x=np.log10(metallicities[mask][final_mask][mask_seeds_4]), y=(weights[mask][final_mask][mask_seeds_4]), c='grey', s=9)
plt.scatter(x=np.log10(metallicities[mask][final_mask][mask_seeds_5]), y=(weights[mask][final_mask][mask_seeds_5]), c='orange', s=9)

axe.set_yscale('log')

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'Sum of Weights'

layoutAxes(ax=axe, nameX=nameX, nameY=nameY, fontsize=15, setMinor=True)
plt.show()"""

##### 
"""for ind, string in enumerate(list_of_models):
    
    mask = mask_data[ind]
    final_mask = get_final_mask(m_BH=mass_bh_data[ind], m_NS=mass_ns_data[ind], mask=mask)
    
    seeds_original = seeds_data[ind]
    weights = weights_data[ind]
    
    masked_weights = weights[mask][final_mask]
    
    mask_seeds_1 = np.in1d(seeds_original, seeds_channels_data[ind][0])
    mask_seeds_2 = np.in1d(seeds_original, seeds_channels_data[ind][1])
    mask_seeds_3 = np.in1d(seeds_original, seeds_channels_data[ind][2])
    mask_seeds_4 = np.in1d(seeds_original, seeds_channels_data[ind][3])
    mask_seeds_5 = np.in1d(seeds_original, seeds_channels_data[ind][4])
    
    print(f'The percentage of weights in the classic channel for model {string} is: {round((np.sum(weights[mask_seeds_1])/np.sum(weights))*100,1)}%')
    print(f'The percentage of weights in the only stable mass transfer channel for model {string} is: {round((np.sum(weights[mask_seeds_2])/np.sum(weights))*100,1)}%')
    print(f'The percentage of weights in the double core CEE channel for model {string} is: {round((np.sum(weights[mask_seeds_3])/np.sum(weights))*100,1)}%')
    print(f'The percentage of weights in the single core CEE channel for model {string} is: {round((np.sum(weights[mask_seeds_4])/np.sum(weights))*100,1)}%')
    print(f'The percentage of weights in the other channels for model {string} is: {round((np.sum(weights[mask_seeds_5])/np.sum(weights))*100,1)}%')
    print("  ")"""
    
##### 

"""ncols, nrows = 1,1
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 40), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

mask = mask_data[4]
final_mask = get_final_mask(m_BH=mass_bh_data[4], m_NS=mass_ns_data[4], mask=mask)

seeds_original = seeds_data[4]
    
mask_seeds_1 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][0])
mask_seeds_2 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][1])
mask_seeds_3 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][2])
mask_seeds_4 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][3])
mask_seeds_5 = np.in1d(seeds_original[mask][final_mask], seeds_channels_data[4][4])

m_bh = mass_bh_data[4]
m_ns = mass_ns_data[4]

plt.scatter(x=(m_ns[mask][final_mask][mask_seeds_1]), y=(m_bh[mask][final_mask][mask_seeds_1]), c='blue', s=4)
plt.scatter(x=(m_ns[mask][final_mask][mask_seeds_2]), y=(m_bh[mask][final_mask][mask_seeds_2]), c='red', s=4)
plt.scatter(x=(m_ns[mask][final_mask][mask_seeds_3]), y=(m_bh[mask][final_mask][mask_seeds_3]), c='yellow', s=4)
plt.scatter(x=(m_ns[mask][final_mask][mask_seeds_4]), y=(m_bh[mask][final_mask][mask_seeds_4]), c='grey', s=4)
plt.scatter(x=(m_ns[mask][final_mask][mask_seeds_5]), y=(m_bh[mask][final_mask][mask_seeds_5]), c='orange', s=4)

nameX = r'$M_{\rm NS} $'
nameY = r'$M_{\rm BH} $'

layoutAxes(ax=axe, nameX=nameX, nameY=nameY, fontsize=15, setMinor=True)
plt.show()"""

##### 

ncols, nrows = 1,1
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,10), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, layout='constrained')

for ind, model in enumerate(list_of_models):
    
    mask_met_BHNS = metallicity_data[ind] <= (0.0142/5)
    
    v_sys = systemic_velocities_data[ind][mask_met_BHNS]
    
    delay_time = delay_times_data[ind][mask_met_BHNS]/1000
    
    weights = weights_data[ind][mask_met_BHNS]
    
    ### Distance Traveled:
    #axe[x[y], n[y]].hist(x=np.log10(xdata), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= 'BHNS', color='black') 
    #axe[x[y], n[y]].hist(x=np.log10(xdata_BNS), bins=len(weights_BNS), histtype='step', density=True, cumulative=True, weights=weights_BNS, label= 'BNS', color='blue')
    
    ### Delay Time:
    
    if not model in ['F', 'G']:
        
        print(f'Starting to plot Model {model}')
        axe.hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='lightgray', alpha=0.5, linewidth=3) 
        print(f'Finished plotting Model {model}')
        
    if model == 'F':
        
        print(f'Starting to plot Model {model}')
        axe.hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= f'Model {model}', color='teal', linewidth=3)
        print(f'Finished plotting Model {model}')
        
    if model == 'G':
        
        print(f'Starting to plot Model {model}')
        axe.hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= f'Model {model}', color='purple', linewidth=3)
        print(f'Finished plotting Model {model}')
        
    ## Systemic Velocity: 
    #axe[x[y], n[y]].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= 'BHNS', color='black') 
    #axe[x[y], n[y]].hist(x=np.log10(v_sys_BNS), bins=len(weights_BNS), histtype='step', density=True, cumulative=True, weights=weights_BNS, label= 'BNS', color='blue')

#plt.axvline(x = 1.6434, color = 'black', linestyle='dashed')
#axe.text(1.67, 0.75, r'$v_{\rm CM} \ \approx \ 44 km/s $' , fontsize=14, verticalalignment='bottom', rotation=90)

plt.axvline(x = 0, color = 'black', linestyle='dashed')
plt.axvline(x = -2.5, color = 'black', linestyle='dashed')

axe.text(0.1, 0.1, r'$t_{\rm merge} \ \approx \ 1 Gyr $' , fontsize=14, verticalalignment='bottom', rotation=90)
axe.text(-2.4, 0.1, r'$t_{\rm merge} \ \approx \ 3.16 Myr $' , fontsize=14, verticalalignment='bottom', rotation=90)

plt.xlim(-6, 1.5)

plt.legend(loc='upper left')

nameX = r'$\log_{{\rm{10}}}(t_{\rm merge}) [Km/s]$'
nameY = r'$CDF$'

layoutAxes(ax=axe, nameX=nameX, nameY=nameY, fontsize=20, setMinor=False)

plt.show()

"""ncols, nrows = 3,6
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 40), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharey=True, sharex=True, layout='constrained')
f.set_constrained_layout_pads(w_pad=2.2, wspace=0.0, hspace=0.0, h_pad=0.08) 

n= [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
x= [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
y=0

for ind, delay_time in enumerate(delay_times_data[1:], 1):
    
    mask = mask_data[ind]
    
    final_mask = get_final_mask(m_BH=mass_bh_data[ind], m_NS=mass_ns_data[ind], mask=mask)
    
    delay_time = delay_time[mask][final_mask]
    
    v_sys = systemic_velocities_data[ind][mask][final_mask]
    
    weights = weights_data[ind][mask][final_mask]
    
    factor = 1.023
    
    xdata = v_sys*delay_time*factor
    
    axe[x[y], n[y]].scatter(x=np.log10(xdata), y=weights, s=1)
    
    y=y+1

nameX = r'$\log_{{\rm{10}}}(Distance\ Traveled) [pc]$'
nameY = r'Weights'
    
f.text(0.5, 0.03, nameX, ha='center', size='large')
f.text(0.02, 0.5, nameY, va='center', rotation='vertical', size='large')

plt.show()"""