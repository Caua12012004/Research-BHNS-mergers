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
from Data_BNS import *

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

ncols, nrows = 2,2
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(26,15), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, layout='constrained', sharex='col', sharey='col')

for ind, model in enumerate(list_of_models):
    
    mask = mask_data_max[ind] 
    
    mask_met_BHNS = metallicity_data[ind] <= (0.0142/5)
    
    v_sys = systemic_velocities_data[ind][mask_met_BHNS]
    
    v_sys_new = systemic_velocities_data[ind][mask]
    
    delay_time = delay_times_data[ind][mask_met_BHNS]/1000
    
    delay_time_new = delay_times_data[ind][mask]/1000
    
    weights = weights_data[ind][mask_met_BHNS]
    
    weights_new = weights_data[ind][mask]
    
    m_bhs = mass_bh_data[ind]
    m_ns = mass_ns_data[ind]
    
    ### Distance Traveled:
    #axe[x[y], n[y]].hist(x=np.log10(xdata), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= 'BHNS', color='black') 
    #axe[x[y], n[y]].hist(x=np.log10(xdata_BNS), bins=len(weights_BNS), histtype='step', density=True, cumulative=True, weights=weights_BNS, label= 'BNS', color='blue')
    
    ### Delay Time:
    
    """if not model in ['A', 'F', 'G']:
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='lightgray', alpha=0.5, linewidth=4.5, zorder = 1000) 
        print(f'Finished plotting Model {model}')"""
    
    if model == 'A':
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='teal', linewidth=4.5, zorder =2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask], color='teal', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'F':
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkslategrey', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask],  color='darkslategrey', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'G':
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='powderblue', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,0].hist(x=np.log10(delay_time_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask], color='powderblue', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
    
    axe[0,0].set_xlim(left=-7.5)
    axe[0,0].set_ylim(0, 1.1)
    
    ## Systemic Velocity: 
    
    """if not model in ['A', 'F', 'G']:
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='lightgray', alpha=0.5, linewidth=4.5, zorder = 1000) 
        print(f'Finished plotting Model {model}')"""
    
    if model == 'A':
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='teal', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask], color='teal', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'F':
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkslategrey', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask],  color='darkslategrey', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'G':
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='powderblue', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        fmask = get_final_mask(m_bhs, m_ns, mask)
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(v_sys_new[fmask]), bins=len(weights_new[fmask]), histtype='step', density=True, cumulative=True, weights=weights_new[fmask], color='powderblue', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
    
    axe[0,1].set_xlim(left=0.25, right=3.0)
    axe[0,1].set_ylim(0, 1.1)
    
    #axe[x[y], n[y]].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= 'BHNS', color='black') 
    #axe[x[y], n[y]].hist(x=np.log10(v_sys_BNS), bins=len(weights_BNS), histtype='step', density=True, cumulative=True, weights=weights_BNS, label= 'BNS', color='blue')


for ind, model in enumerate(list_of_models):
    
    
    
    if model == 'A' or model == "F" or model == "C":
        
        mask_met_BNS = metallicity_data_BNS[ind] <= (0.0142/5)
        
        v_sys = systemic_velocities_data_BNS[ind][mask_met_BNS]
        
        delay_time = delay_times_data_BNS[ind][mask_met_BNS]/1000
        
        weights = weights_data_BNS[ind][mask_met_BNS]
        
        v_sys_new = systemic_velocities_data_BNS[ind]
        
        delay_time_new = delay_times_data_BNS[ind]/1000
        
        weights_new = weights_data_BNS[ind]
        
        f_mask = mask_data_BNS[ind]
    
    ### Distance Traveled:
    #axe[x[y], n[y]].hist(x=np.log10(xdata), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, label= 'BHNS', color='black') 
    #axe[x[y], n[y]].hist(x=np.log10(xdata_BNS), bins=len(weights_BNS), histtype='step', density=True, cumulative=True, weights=weights_BNS, label= 'BNS', color='blue')
    
    ### Delay Time:
    
    """if not model in ['A', 'F', 'G']:
        
        print(f'Starting to plot Model {model}')
        axe[0,1].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='lightgray', alpha=0.5, linewidth=4.5, zorder = 1000) 
        print(f'Finished plotting Model {model}')"""
    
    if model == 'A':
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='goldenrod', linewidth=4.5, zorder =2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='goldenrod', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'F':
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkgoldenrod', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='darkgoldenrod', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'C':
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkkhaki', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,0].hist(x=np.log10(delay_time_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='darkkhaki', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
    
    axe[1,0].set_xlim(left=-7.5)
    axe[1,0].set_ylim(0, 1.1)
    
    ## Systemic Velocity: 
    
    """if not model in ['A', 'F', 'G']:
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='lightgray', alpha=0.5, linewidth=4.5, zorder = 1000) 
        print(f'Finished plotting Model {model}')"""
    
    if model == 'A':
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights,  color='goldenrod', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='goldenrod', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'F':
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkgoldenrod', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='darkgoldenrod', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
        
    if model == 'C':
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys), bins=len(weights), histtype='step', density=True, cumulative=True, weights=weights, color='darkkhaki', linewidth=4.5, zorder = 2000)
        print(f'Finished plotting Model {model}')
        
        print(f'Starting to plot Model {model}')
        axe[1,1].hist(x=np.log10(v_sys_new[f_mask]), bins=len(weights_new[f_mask]), histtype='step', density=True, cumulative=True, weights=weights_new[f_mask], color='darkkhaki', linewidth=4.5, ls = '--', zorder =2000)
        print(f'Finished plotting Model {model}')
    
    axe[1,1].set_xlim(left=0.25, right = 3.0)
    axe[1,1].set_ylim(0, 1.1)
    
axe[1,1].axvline(x = 1.6434, color = 'black', linestyle='dashed', linewidth=4.5, zorder=3000)
axe[1,1].text(1.68, 0.6, r'$v_{\rm CM} \ \approx \: 44 \rm{km/s} $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)

axe[0,1].axvline(x = 1.6434, color = 'black', linestyle='dashed', linewidth=4.5, zorder=3000)
axe[0,1].text(1.68, 0.45, r'$v_{\rm CM} \ \approx \: 44 {km/s} $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)

axe[0,0].axvline(x = 0, color = 'black', linestyle='dashed', zorder=3000, linewidth=4.5)
axe[0,0].axvline(x = -2, color = 'black', linestyle='dashed', zorder=3000, linewidth=4.5)

axe[0,0].text(0.1, 0.08, r'$t_{\rm merge} \ \approx \ 1 Gyr $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)
axe[0,0].text(-2.3, 0.3, r'$t_{\rm merge} \ \approx \ 10 Myr $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)


axe[1,0].axvline(x = 0, color = 'black', linestyle='dashed', zorder=3000, linewidth=4.5)
axe[1,0].axvline(x = -2, color = 'black', linestyle='dashed', zorder=3000, linewidth=4.5)

axe[1,0].text(0.1, 0.1, r'$t_{\rm merge} \ \approx \ 1 Gyr $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)
axe[1,0].text(-1.92, 0.45, r'$t_{\rm merge} \ \approx \ 10 Myr $' , fontsize=25, verticalalignment='bottom', rotation=90, zorder=3000)

#axe[0,0].legend(loc='upper left')
#axe[0,1].legend(loc='upper left')
#axe[1,0].legend(loc='upper left')
#axe[1,1].legend(loc='upper left')

nameX = r'$\log_{{\rm{10}}}(t_{\rm merge}) [Gyr]$'
nameY = r'$\rm{CDF}$'

nameX_1= r'$\log_{{\rm{10}}}(v_{\rm CM}) [\rm{km/s}]$'

layoutAxes(ax=axe[0,0], nameX=nameX, nameY=nameY, fontsize=25, setMinor=True, labelpad=13)
layoutAxes(ax=axe[1,0], nameX=nameX, nameY=nameY, fontsize=25, setMinor=True)

layoutAxes(ax=axe[0,1], nameX=nameX_1, nameY=nameY, fontsize=25, setMinor=True, labelpad=13)
layoutAxes(ax=axe[1,1], nameX=nameX_1, nameY=nameY, fontsize=25, setMinor=True)

### LABEL STUFF: 

axe[0,0].plot([-1,0], [-1,-1], c='teal', label='Model F')
axe[0,0].plot([-1,0], [-1,-1], c='darkslategrey', label='Model A')
axe[0,0].plot([-1,0], [-1,-1], c='powderblue', label='Model G')
axe[0,0].plot([-1,0], [-1,-1], c='black', label='All Systems')
axe[0,0].plot([-1,0], [-1,-1], c='black', ls='--', label='Candidate Systems')

axe[0,1].plot([-1,0], [-1,-1], c='teal', label='Model F')
axe[0,1].plot([-1,0], [-1,-1], c='darkslategrey', label='Model A')
axe[0,1].plot([-1,0], [-1,-1], c='powderblue', label='Model G')
axe[0,1].plot([-1,0], [-1,-1], c='black', label='All Systems')
axe[0,1].plot([-1,0], [-1,-1], c='black', ls='--', label='Candidate Systems')

axe[1,0].plot([-1,0], [-1,-1], c='darkgoldenrod', label='Model F')
axe[1,0].plot([-1,0], [-1,-1], c='goldenrod', label='Model A')
axe[1,0].plot([-1,0], [-1,-1], c='darkkhaki', label='Model C')
axe[1,0].plot([-1,0], [-1,-1], c='black', label='All Systems')
axe[1,0].plot([-1,0], [-1,-1], c='black', ls='--', label='Candidate Systems')

axe[1,1].plot([-1,0], [-1,-1], c='darkgoldenrod', label='Model F')
axe[1,1].plot([-1,0], [-1,-1], c='goldenrod', label='Model A')
axe[1,1].plot([-1,0], [-1,-1], c='darkkhaki', label='Model C')
axe[1,1].plot([-1,0], [-1,-1], c='black', label='All Systems')
axe[1,1].plot([-1,0], [-1,-1], c='black', ls='--', label='Candidate Systems')

axe[0,0].legend(loc='upper left', title='BHNS', alignment='center', title_fontsize='xx-large', fontsize=22)
axe[0,1].legend(loc='upper left', title='BHNS', alignment='center', title_fontsize='xx-large', fontsize=22)
axe[1,0].legend(loc='upper left', title='BNS', alignment='center', title_fontsize='xx-large', fontsize=22)
axe[1,1].legend(loc='upper left', title='BNS', alignment='center', title_fontsize='xx-large', fontsize=22)

axe[0,0].grid(alpha=0.5)
axe[1,0].grid(alpha=0.5)
axe[0,1].grid(alpha=0.5)
axe[1,1].grid(alpha=0.5)

plt.savefig('./CDF_general_test_4.pdf', bbox_inches='tight', dpi = 600)

plt.show()