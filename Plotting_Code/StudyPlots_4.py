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
from Data_Colombo import *
import EOS_properties as ep
import m_ejecta_disk_lambda as medl
import PostProcessingScripts
from Data_BNS import *
from PostProcessingScripts import *

from Interpolation_BHNS import x_separation_BHNS_1
from Interpolation_BHNS import x_separation_BHNS_2

from Interpolation_BHNS import x_eccentricities_BHNS_1
from Interpolation_BHNS import x_eccentricities_BHNS_2

from Interpolation_BNS import x_separation_BNS_1
from Interpolation_BNS import x_separation_BNS_2

from Interpolation_BNS import x_eccentricities_BNS_1
from Interpolation_BNS import x_eccentricities_BNS_2

AUtoRsun = 214.939

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

def get_final_mask_all(m_BH, m_NS):
    
    Mbh = m_BH
    Mns = m_NS
    
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
    
    final_mask_all = Mej_list > 0
    
    return final_mask_all


models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model F', 'Model G', 'Model H', 'Model I', 'Model J', 'Model K', 'Model L', 'Model M', 'Model N', 'Model O', 'Model P', 'Model Q', 'Model R', 'Model S', 'Model T']

ncols, nrows = 2,2
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(20, 15), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, layout='constrained')

### BHNS first plot: 

mask_met_BHNS = metallicity_data[0] <= (0.0142/5)

mask_BHNS = mask_data_max[0]

axe[0,0].scatter(x=np.log10(systemic_velocities_data[0][mask_met_BHNS]), y=np.log10(delay_times_data[0][mask_met_BHNS]), c='lightgrey', s=9)

final_mask_BHNS = get_final_mask(m_BH=mass_bh_data[0], m_NS=mass_ns_data[0], mask=mask_BHNS)

final_mask_BHNS_all = get_final_mask_all(m_BH=mass_bh_data[0][mask_met_BHNS], m_NS=mass_ns_data[0][mask_met_BHNS])

systemic_velocities_BHNS = systemic_velocities_data[0]
delay_times_BHNS = delay_times_data[0]

axe[0,0].scatter(x=np.log10(systemic_velocities_BHNS[mask_met_BHNS][final_mask_BHNS_all]), y=np.log10(delay_times_BHNS[mask_met_BHNS][final_mask_BHNS_all]), c='darkturquoise', s=9, label='Almost Candidate BHNS')

axe[0,0].scatter(x=np.log10(systemic_velocities_BHNS[mask_BHNS][final_mask_BHNS]), y=np.log10(delay_times_BHNS[mask_BHNS][final_mask_BHNS]), c='darkslategrey', s=9, label='Candidate BHNS')

axe[0,0].legend(loc='lower left', fontsize = 'xx-large')

axe[0,0].axvline(x=1.6434, ymin=0, ymax=1, color="black", linestyle="--")
axe[0,0].axhline(y=3, xmin=0, xmax=1, color="black", linestyle="--")
axe[0,0].axhline(y=0.5, xmin=0, xmax=1, color="black", linestyle="--")

axe[0,0].text(1.67, -11, r'$v_{\rm CM} \ \approx \ 44 km/s $' , fontsize=14, rotation=90, verticalalignment='bottom')
axe[0,0].text(-0.4, 3.75, r'$t_{\rm merge} \ \approx \ 1 Gyr $' , fontsize=14, verticalalignment='top')
axe[0,0].text(-0.4, 1.25, r'$t_{\rm merge} \ \approx \ 3.16 Myr $' , fontsize=14, verticalalignment='top')

nameX = r'$\log_{{\rm{10}}}(v)$$[km/s]$'
nameY = r'$\log_{{\rm{10}}}(t_{\rm merge})$$[Myr]$'

axe[0,0].set_ylim(np.min(np.log10(delay_times_data[0][mask_met_BHNS])), np.max(np.log10(delay_times_data[0][mask_met_BHNS])))

layoutAxes(ax=axe[0,0], nameX=nameX, nameY=nameY, fontsize=20, setMinor=True)

### BHNS second plot:

axe[1, 0].scatter(x=np.log10(separations_data[0][mask_met_BHNS]*AUtoRsun), y=eccentricities_data[0][mask_met_BHNS], c='lightgrey', s=9)

eccentricities_BHNS = eccentricities_data[0]
separations_BHNS = separations_data[0]

axe[1,0].scatter(x=np.log10(separations_BHNS[mask_met_BHNS][final_mask_BHNS_all]*AUtoRsun), y=eccentricities_BHNS[mask_met_BHNS][final_mask_BHNS_all], c='darkturquoise', s=9, label='Almost Candidate BHNS')

axe[1,0].scatter(x=np.log10(separations_BHNS[mask_BHNS][final_mask_BHNS]*AUtoRsun), y=eccentricities_BHNS[mask_BHNS][final_mask_BHNS], c='darkslategrey', s=9, label='Candidate BHNS')

axe[1,0].legend(loc='lower left', fontsize = 'xx-large')

axe[1,0].plot(np.log10((x_separation_BHNS_1.value)), x_eccentricities_BHNS_1, lw=4, color = 'k',linestyle=':', zorder=2000)
axe[1,0].plot(np.log10((x_separation_BHNS_2.value)), x_eccentricities_BHNS_2, lw=4, color = 'k',linestyle=':', zorder=2000)

axe[1,0].set_ylim(0, 1)

strr = r"$t_{\rm{inspiral}}=3.16\,\rm{Myr}$"
x_xy, y_xy = 0.37, 0.55 #0.31,0.7
x_xytext, y_xytext = 0.22, 0.75 
conn = "arc3,rad=+0.2"

axe[1,0].annotate(strr,
              xy=(x_xy, y_xy), xycoords='axes fraction',
              xytext=(x_xytext, y_xytext), textcoords='axes fraction',
              size=14, va="center", ha="center",
        #                   bbox=dict(boxstyle="round4", fc="w"),
              arrowprops=dict(arrowstyle="-|>",
                              connectionstyle=conn,
                              fc="k", lw=2.5), zorder=20000
                          )   

strr_1 = r"$t_{\rm{inspiral}}=1\,\rm{Gyr}$"
x_xy_1, y_xy_1 = 0.48, 0.6
x_xytext_1, y_xytext_1 = 0.82, 0.7
conn_1 = "arc3,rad=-0.2"

axe[1,0].annotate(strr_1,
              xy=(x_xy_1, y_xy_1), xycoords='axes fraction',
              xytext=(x_xytext_1, y_xytext_1), textcoords='axes fraction',
              size=14, va="center", ha="center",
        #                   bbox=dict(boxstyle="round4", fc="w"),
              arrowprops=dict(arrowstyle="-|>",
                              connectionstyle=conn_1,
                              fc="k", lw=2.5), zorder=20000
                          ) 

nameX_1 = r'$\log_{{\rm{10}}}(a)$$[R_{\odot}]$'
nameY_1 = 'Eccentricity'

layoutAxes(ax=axe[1,0], nameX=nameX_1, nameY=nameY_1, fontsize=20, setMinor=True, labelpad=12)

### BNS first plot: 

mask_met_BNS = metallicity_data_BNS[0] <= (0.0142/5)

mask_BNS = mask_data_BNS[0]
axe[0,1].scatter(x=np.log10(systemic_velocities_data_BNS[0][mask_met_BNS]), y=np.log10(delay_times_data_BNS[0][mask_met_BNS]), c='lightgrey', s=9)

systemic_velocities_BNS = systemic_velocities_data_BNS[0]
delay_times_BNS = delay_times_data_BNS[0]

axe[0,1].scatter(x=np.log10(systemic_velocities_BNS[mask_BNS]), y=np.log10(delay_times_BNS[mask_BNS]), c='gold', s=9, label='Candidate BNS')

axe[0,1].legend(loc='lower right', fontsize = 'xx-large')

axe[0,1].axvline(x=1.6434, ymin=0, ymax=1, color="black", linestyle="--")
axe[0,1].axhline(y=3, xmin=0, xmax=1, color="black", linestyle="--")
axe[0,1].axhline(y=0.5, xmin=0, xmax=1, color="black", linestyle="--")

axe[0,1].text(1.67, -9, r'$v_{\rm CM} \ \approx \ 44 km/s $' , fontsize=14, rotation=90, verticalalignment='bottom')
axe[0,1].text(-0.3, 3.75, r'$t_{\rm merge} \ \approx \ 1 Gyr $' , fontsize=14, verticalalignment='top')
axe[0,1].text(-0.3, 1.25, r'$t_{\rm merge} \ \approx \ 3.16 Myr $' , fontsize=14, verticalalignment='top')

axe[0,1].set_ylim(np.min(np.log10(delay_times_data_BNS[0][mask_met_BNS])), np.max(np.log10(delay_times_data_BNS[0][mask_met_BNS])))

layoutAxes(ax=axe[0,1], nameX=nameX, nameY=nameY, fontsize=20, setMinor=True)

### BNS second plot: 

axe[1, 1].scatter(x=np.log10(separations_data_BNS[0][mask_met_BNS]*AUtoRsun), y=eccentricities_data_BNS[0][mask_met_BNS], c='lightgrey', s=9)

eccentricities_BNS = eccentricities_data_BNS[0]
separations_BNS = separations_data_BNS[0]

axe[1,1].scatter(x=np.log10(separations_BNS[mask_BNS]*AUtoRsun), y=eccentricities_BNS[mask_BNS], c='gold', s=9, label='Candidate BNS')

axe[1,1].legend(loc='lower right', fontsize = 'xx-large')

axe[1,1].plot(np.log10((x_separation_BNS_1.value)), x_eccentricities_BNS_1, lw=4, color = 'k',linestyle=':', zorder=2000)
axe[1,1].plot(np.log10((x_separation_BNS_2.value)), x_eccentricities_BNS_2, lw=4, color = 'k',linestyle=':', zorder=2000)

axe[1,1].set_ylim(0, 1)

layoutAxes(ax=axe[1,1], nameX=nameX_1, nameY=nameY_1, fontsize=20, setMinor=True, labelpad=12)

strr_2 = r"$t_{\rm{inspiral}}=1\,\rm{Gyr}$"
x_xy_2, y_xy_2 = 0.34, 0.6 #0.48, 0.6
x_xytext_2, y_xytext_2 = 0.64, 0.7
conn_2 = "arc3,rad=-0.2"

axe[1,1].annotate(strr_2,
              xy=(x_xy_2, y_xy_2), xycoords='axes fraction',
              xytext=(x_xytext_2, y_xytext_2), textcoords='axes fraction',
              size=14, va="center", ha="center",
        #                   bbox=dict(boxstyle="round4", fc="w"),
              arrowprops=dict(arrowstyle="-|>",
                              connectionstyle=conn_2,
                              fc="k", lw=2.5), zorder=20000
                          ) 

strr_3 = r"$t_{\rm{inspiral}}=3.16\,\rm{Myr}$"
x_xy_3, y_xy_3 = 0.24, 0.75 #0.31,0.7
x_xytext_3, y_xytext_3 = 0.17, 0.95 
conn_3 = "arc3,rad=+0.2"

axe[1,1].annotate(strr_3,
              xy=(x_xy_3, y_xy_3), xycoords='axes fraction',
              xytext=(x_xytext_3, y_xytext_3), textcoords='axes fraction',
              size=14, va="center", ha="center",
        #                   bbox=dict(boxstyle="round4", fc="w"),
              arrowprops=dict(arrowstyle="-|>",
                              connectionstyle=conn_3,
                              fc="k", lw=2.5), zorder=20000
                          )   

plt.savefig('./ResearchFig1_test_2.png', bbox_inches='tight', dpi = 600)

plt.show()
plt.close()