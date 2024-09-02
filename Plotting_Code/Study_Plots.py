#import sys
# sys.path.append('../Scripts')
# from PostProcessingScripts import * 
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
from Data_BNS import *
import PostProcessingScripts

met = metallicity_data_BNS[0]
mask_test = met <= 0.0142/5

print(np.unique(met[mask_test]))
print(len(np.unique(met[mask_test])))

## DELAY TIME VS SYSTEMIC VELOCITY PLOT: 

"""def get_final_mask(m_BH, m_NS, mask):
    
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
    
    return final_mask"""


models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model F', 'Model G', 'Model H', 'Model I', 'Model J', 'Model K', 'Model L', 'Model M', 'Model N', 'Model O', 'Model P', 'Model Q', 'Model R', 'Model S', 'Model T']

### SYSTEMIC VELOCITY VS MERGER TIME PLOT: 

"""ncols, nrows = 3,6
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 10), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharey=True, sharex=True, layout='constrained')
f.set_constrained_layout_pads(w_pad=2.2, wspace=0.0, hspace=0.0, h_pad=0.08) 

n= [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
x= [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
y=0

for ind, delay_time in enumerate(delay_times_data[1:], 1):
    
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities_data[ind]), y=np.log10(delay_time), c='grey', s=1)
    
    mask = mask_data[ind]
    final_mask = get_final_mask(m_BH=mass_bh_data[ind], m_NS=mass_ns_data[ind], mask=mask)
    
    systemic_velocities = systemic_velocities_data[ind]
    axe[x[y], n[y]].scatter(x=np.log10(systemic_velocities[mask][final_mask]), y=np.log10(delay_time[mask][final_mask]), c='teal', s=1)
    
    axe[x[y], n[y]].text(0.05, 0.1, models[ind], horizontalalignment='left', verticalalignment='bottom', transform=axe[x[y], n[y]].transAxes)
    
    if not models[ind] in ['Model R', 'Model S', 'Model T']:
        axe[x[y], n[y]].tick_params(axis='x', which='both', bottom=False)
    
    if not models[ind] in ['Model B', 'Model F', 'Model I', 'Model L', 'Model O', 'Model R']:
        axe[x[y], n[y]].tick_params(axis='y', which='both', left=False)
    
    y=y+1

nameX = r'$\log_{{\rm{10}}}(v)$$[km/s]$'
nameY = r'$\log_{{\rm{10}}}(t_{merge})$$[Myr]$'
    
f.text(0.5, 0.03, nameX, ha='center', size='x-large')
f.text(0.02, 0.5, nameY, va='center', rotation='vertical', size='x-large')

plt.savefig('./ResearchFig2.png', bbox_inches='tight', dpi = 600)

plt.show()
plt.close()"""

### SEMI-MAJOR AXIS VS ECCENTRICITY PLOT: 

ncols, nrows = 3,6
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(10, 10), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharey=True, sharex=True, layout='constrained')
f.set_constrained_layout_pads(w_pad=2.2, wspace=0.0, hspace=0.0, h_pad=0.08) 

n= [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]
x= [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
y=0

AUtoRsun = 214.939


for ind, separations in enumerate(separations_data_BNS[1:], 1):
    
    axe[x[y], n[y]].scatter(x=np.log10(separations*AUtoRsun), y=eccentricities_data_BNS[ind], c='grey', s=1)
    
    mask = mask_data_BNS[ind]
    
    eccentricities = eccentricities_data_BNS[ind]
    axe[x[y], n[y]].scatter(x=np.log10(separations[mask]*AUtoRsun), y=eccentricities[mask], c='teal', s=1)
    
    print(len(separations[mask]))
    
    axe[x[y], n[y]].text(0.6, 0.1, models[ind], horizontalalignment='left', verticalalignment='bottom', transform=axe[x[y], n[y]].transAxes)
    
    if not models[ind] in ['Model R', 'Model S', 'Model T']:
        axe[x[y], n[y]].tick_params(axis='x', which='both', bottom=False)
    
    if not models[ind] in ['Model B', 'Model F', 'Model I', 'Model L', 'Model O', 'Model R']:
        axe[x[y], n[y]].tick_params(axis='y', which='both', left=False)
    
    y=y+1

nameX = r'$\log_{{\rm{10}}}(a)$$[R_{\odot}]$'
nameY = 'Eccentricity'
    
f.text(0.5, 0.03, nameX, ha='center', size='x-large')
f.text(0.02, 0.5, nameY, va='center', rotation='vertical', size='x-large')

plt.savefig('./ResearchFig3_BNS.png', bbox_inches='tight', dpi = 600)

plt.show()
plt.close()