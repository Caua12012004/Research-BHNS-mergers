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
from Data_BNS import Rates_data
from Data_BNS import err
from Data_BNS import y_positions

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

yerror = [err, err*0]

nameX = r'$\log_{{\rm{10}}}(Z) $'
nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

x_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

dic_xlabels = {10:r'\textbf{fiducial}', 20:r'$\beta=0.25$', 30:r'$\beta=0.5$',  40:r'$\beta=0.75$', 50:r'\textbf{E + K}', 60:r'$\alpha_{\rm{CE}}=0.1$', 70:r'$\alpha_{\rm{CE}}=0.5$',  80:r'$\alpha_{\rm{CE}}=2$', 90:r'$\alpha_{\rm{CE}}=10$', 100: r'\textbf{optimistic}' +'\n' + r'\textbf{CE}', \
               110:r'\textbf{rapid SN}', 120:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$2.0\,\rm{M}_{\odot}$', 130:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$3.0\,\rm{M}_{\odot}$', \
                140:r'\textbf{no PISN}', 150:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$100\,\rm{km}\,\rm{s}^{-1}$', 160:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$30\,\rm{km}\,\rm{s}^{-1}$', \
                170:r'\textbf{SN} '+ r'$v_{\rm{k,BH}}$' +'\n' + r'$0\,\rm{km}\,\rm{s}^{-1}$', 180:r'$\rm{f}_{\rm{WR}} = 0.1$', 190:r'$\rm{f}_{\rm{WR}} = 5$' }

"""ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='gold', alpha=0.6, lw=0)

n=0
for rates in Rates_data:
    x_array = np.full_like(rates, x_ticks[n])
    axe.scatter(x=x_array, y=rates, c='gold', s=16)
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

### BAR PLOT: 

ncols, nrows = 1,1 
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(15,9), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)})

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='gold', alpha=0.6, lw=0)

plt.errorbar(x=x_ticks,y=y_positions, yerr=yerror, marker = '.',  fmt='none', elinewidth=4, c='gold', label='Candidate BNS rates')

axe.text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=15)
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
plt.show()