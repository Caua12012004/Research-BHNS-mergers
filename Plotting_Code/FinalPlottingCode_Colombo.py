#import sys
# sys.path.append('../Scripts')
# from PostProcessingScripts import * 
# import pandas as pd 
# import string 
# just to make the cells appear wider:
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from PostProcessingScripts import layoutAxesNoXlabel
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
from Data_Colombo import Rates_data
from Data_Colombo import err
from Data_Colombo import y_positions
#from Data_BNS import err_BNS
#from Data_BNS import y_positions_BNS

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

x_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

yerror = [err, err*0]

dic_xlabels = {10:r'\textbf{fiducial}', 20:r'$\beta=0.25$', 30:r'$\beta=0.5$',  40:r'$\beta=0.75$', 50:r'\textbf{E + K}', 60:r'$\alpha_{\rm{CE}}=0.1$', 70:r'$\alpha_{\rm{CE}}=0.5$',  80:r'$\alpha_{\rm{CE}}=2$', 90:r'$\alpha_{\rm{CE}}=10$', 100: r'\textbf{optimistic}' +'\n' + r'\textbf{CE}', \
               110:r'\textbf{rapid SN}', 120:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$2.0\,\rm{M}_{\odot}$', 130:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$3.0\,\rm{M}_{\odot}$', \
                140:r'\textbf{no PISN}', 150:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$100\,\rm{km}\,\rm{s}^{-1}$', 160:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$30\,\rm{km}\,\rm{s}^{-1}$', \
                170:r'\textbf{SN} '+ r'$v_{\rm{k,BH}}$' +'\n' + r'$0\,\rm{km}\,\rm{s}^{-1}$', 180:r'$\rm{f}_{\rm{WR}} = 0.1$', 190:r'$\rm{f}_{\rm{WR}} = 5$' }

ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

plt.errorbar(x=x_ticks,y=y_positions, yerr=yerror, marker = '.',  fmt='none', elinewidth=4, c='teal', label='Candidate BHNS rates')

axe.text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
axe.text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

#plt.arrow(91,y_positions[8],4,0,width=10e-8, length_includes_head=True)

plt.ylim((10e-9, 10e-5))

axe.set_yscale('log')
layoutAxesNoXlabel(ax=axe, nameY=nameY, fontsize=15, setMinor=False)
axe.set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe.legend(loc='upper left', fontsize='x-large')
axe.yaxis.set_label_coords(-0.07, 0.5)
axe.grid()
plt.show()

## 11 Km + 0 spin case: 

"""ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

max_rates = np.asarray([])
min_rates = np.asarray([])

for rates in Rates_data:
    try:
        
        max_rate = np.max(rates)
        min_rate = np.min(rates)
    
    except ValueError:
        
        max_rate = 0
        min_rate = 0 
    
    max_rates = np.append(max_rates, max_rate)
    min_rates = np.append(min_rates, min_rate)

yerror = [np.subtract(max_rates, min_rates), np.subtract(max_rates, min_rates)*0]

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

plt.errorbar(x=x_ticks,y=max_rates, yerr=yerror, marker = '.',  fmt='none', elinewidth=4, c='teal', label='Candidate BHNS rates')

axe.text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)d
axe.text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

plt.ylim((10e-9, 10e-5))

axe.set_yscale('log')
layoutAxesNoXlabel(ax=axe, nameY=nameY, fontsize=15, setMinor=False)
axe.set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe.legend(loc='upper left', fontsize='x-large')
axe.yaxis.set_label_coords(-0.07, 0.5)
axe.grid()
plt.show()"""



#### ANOTHER OPTION: 

"""ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

n=0
for rates in Rates_data:
    x_array = np.full_like(rates, x_ticks[n])
    axe.scatter(x=x_array, y=rates, c='teal', s=16)
    n = n + 1 

axe.text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
axe.text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

plt.ylim((10e-9, 10e-5))

axe.set_yscale('log')
layoutAxesNoXlabel(ax=axe, nameY=nameY, fontsize=15, setMinor=False)
axe.set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe.legend(loc='lower left', fontsize='x-large')
axe.yaxis.set_label_coords(-0.07, 0.5)
axe.set_axisbelow(True)
axe.grid()
plt.show()"""

### BHNS vs BNS case: 

"""ncols, nrows = 1,2
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(30,18), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, layout='constrained')

axe[0].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe[0].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

axe[0].errorbar(x=x_ticks,y=y_positions, yerr=yerror, marker = '.',  fmt='none', elinewidth=4, c='teal', label='Candidate BHNS rates')

axe[0].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
axe[0].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

axe[0].set_ylim((10e-9, 10e-5))

axe[0].set_yscale('log')

layoutAxesNoXlabel(ax=axe[0], nameY=nameY, fontsize=15, setMinor=False)

axe[0].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe[0].legend(loc='upper right', fontsize='x-large')
axe[0].yaxis.set_label_coords(-0.07, 0.5)
axe[0].grid()

yerror_BNS = [err_BNS, err_BNS*0]

axe[1].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe[1].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='gold', alpha=0.6, lw=0)

axe[1].errorbar(x=x_ticks,y=y_positions_BNS, yerr=yerror_BNS, marker = '.',  fmt='none', elinewidth=4, c='gold', label='Candidate BNS rates')

axe[1].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
axe[1].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

axe[1].set_ylim((10e-9, 10e-5))

axe[1].set_yscale('log')
layoutAxesNoXlabel(ax=axe[1], nameY=nameY, fontsize=15, setMinor=False)
axe[1].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe[1].legend(loc='upper right', fontsize='x-large')
axe[1].yaxis.set_label_coords(-0.07, 0.5)
axe[1].grid()


plt.show()"""

### ANOTHER OPTION:

"""ncols, nrows = 1,1
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

axe.text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
axe.text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

x_ticks_1 = [9.5, 19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5, 99.5, 109.5, 119.5, 129.5, 139.5, 149.5, 159.5, 169.5, 179.5, 189.5]

plt.errorbar(x=x_ticks_1,y=y_positions, yerr=yerror, marker = '.',  fmt='none', elinewidth=4, c='teal', label='Candidate BHNS rates')

yerror_BNS = [err_BNS, err_BNS*0]

x_ticks_2 = [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5, 100.5, 110.5, 120.5, 130.5, 140.5, 150.5, 160.5, 170.5, 180.5, 190.5]

plt.errorbar(x=x_ticks_2,y=y_positions_BNS, yerr=yerror_BNS, marker = '.',  fmt='none', elinewidth=4, c='gold', alpha=0.6, label='Candidate BNS rates')

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

plt.ylim((10e-9, 10e-5))

plt.scatter(x=[90.5, 90.5], y=[y_positions_BNS[8]-err_BNS[8], y_positions_BNS[8]], c='gold')
    
axe.set_yscale('log')
layoutAxesNoXlabel(ax=axe, nameY=nameY, fontsize=15, setMinor=False)
axe.set_xticks(ticks=x_ticks, labels=xlabels, fontsize=8, rotation=30)
axe.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe.legend(loc='upper left', fontsize='x-large', ncols=2)
axe.yaxis.set_label_coords(-0.07, 0.5)
axe.grid()


plt.show()"""