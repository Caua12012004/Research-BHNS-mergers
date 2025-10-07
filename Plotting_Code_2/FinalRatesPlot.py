from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from PostProcessingScripts import layoutAxesNoXlabel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from Data_Colombo import *
from Data_BNS import* 

Max_Reticulum_II_Mass = 2.8*(10**4)
Min_Reticulum_II_Mass = 2.4*(10**4)

Max_Tucana_III_Mass = 3.2*(10**4)
Min_Tucana_III_Mass = 18.4*(10**4)

nameY = r'$\mathcal{R}_{\rm{form}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

Z_solar = 0.0142

dic_xlabels = {10:r'\textbf{fiducial}', 20:r'$\beta=0.25$', 30:r'$\beta=0.5$',  40:r'$\beta=0.75$', 50:r'\textbf{E + K}', 60:r'$\alpha_{\rm{CE}}=0.1$', 70:r'$\alpha_{\rm{CE}}=0.5$',  80:r'$\alpha_{\rm{CE}}=2$', 90:r'$\alpha_{\rm{CE}}=10$', 100: r'\textbf{optimistic}' +'\n' + r'\textbf{CE}', \
               110:r'\textbf{rapid SN}', 120:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$2.0\,\rm{M}_{\odot}$', 130:r'$\rm{max} \ m_{\rm{NS}}$' +'\n' + r'$3.0\,\rm{M}_{\odot}$', \
                140:r'\textbf{no PISN}', 150:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$100\,\rm{km}\,\rm{s}^{-1}$', 160:r'\textbf{SN} '+ r'$\sigma_{\rm{rms}}^{\rm{1D}}$' +'\n' + r'$30\,\rm{km}\,\rm{s}^{-1}$', \
                170:r'\textbf{SN} '+ r'$v_{\rm{k,BH}}$' +'\n' + r'$0\,\rm{km}\,\rm{s}^{-1}$', 180:r'$\rm{f}_{\rm{WR}} = 0.1$', 190:r'$\rm{f}_{\rm{WR}} = 5$' }

x_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

x_ticks_1 = [9.5 - 0.75, 19.5 - 0.75, 29.5 - 0.75, 39.5 - 0.75, 49.5 - 0.75, 59.5 - 0.75, 69.5 - 0.75, 79.5  - 0.75,
             89.5  - 0.75, 99.5  - 0.75, 109.5  - 0.75, 119.5  - 0.75, 129.5  - 0.75, 139.5  - 0.75, 149.5  - 0.75, 159.5  - 0.75, 
             169.5  - 0.75, 179.5  - 0.75, 189.5  - 0.75]

x_ticks_2 = [10.5  + 0.75, 20.5  + 0.75, 30.5  + 0.75, 40.5  + 0.75, 50.5  + 0.75, 60.5  + 0.75, 70.5  + 0.75, 80.5  + 0.75, 
             90.5  + 0.75, 100.5  + 0.75, 110.5  + 0.75, 120.5  + 0.75, 130.5  + 0.75, 140.5  + 0.75, 150.5  + 0.75, 160.5  + 0.75, 
             170.5  + 0.75, 180.5  + 0.75, 190.5  + 0.75]

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]


### TO GET METALLICITY GROUP:

def met_group(met):
    
    if met <= Z_solar/5 and met > Z_solar/10:
        return "fifth"
    
    elif met <= Z_solar/10 and met > Z_solar/20:
        return "tenth"
    
    elif met <= Z_solar/20 and met > Z_solar/50:
        return "twentieth"
    
    elif met <= Z_solar/50:
        return "fifty"
    
### TO GET BNS DATA POINTS:

data_points_fifth_max_BNS = []
data_points_tenth_max_BNS = []
data_points_twentieth_max_BNS = []
data_points_fifty_max_BNS = []

#gm_1 = []
#gm_2 = []
#gm_3 = []
#gm_4 = []

data_points_fifth_min_BNS = []
data_points_tenth_min_BNS = []
data_points_twentieth_min_BNS = []
data_points_fifty_min_BNS = []

#gM_1 = []
#gM_2 = []
#gM_3 = []
#gM_4 = []

for ind, model in enumerate(list_of_models):
    
    gm_1 = []
    gm_2 = []
    gm_3 = []
    gm_4 = []
    
    gM_1 = []
    gM_2 = []
    gM_3 = []
    gM_4 = []
    
    metallicity = metallicity_data_BNS[ind]
    
    mask_max = mask_data_max_BNS[ind]
    mask_min = mask_data_max_BNS[ind]
    
    rates_max = model_rates_max_BNS[ind]
    rates_min = model_rates_min_BNS[ind]
    
    print(f"Sizes: {np.unique(metallicity[mask_max])} - {len(rates_max)}")
    
    for met, rate in zip(np.unique(metallicity[mask_max]), rates_max):
        
        if met_group(met) == "fifth":
            gM_1 = np.append(gM_1, rate)
        
        elif met_group(met) == "tenth":
            gM_2 = np.append(gM_2, rate)
        
        elif met_group(met) == "twentieth":
            gM_3 = np.append(gM_3, rate)
        
        elif met_group(met) == "fifty":
            gM_4 = np.append(gM_4, rate)
            
        
    for met, rate in zip(np.unique(metallicity[mask_min]), rates_min):
        
        if met_group(met) == "fifth":
            gm_1 = np.append(gm_1, rate)
        
        elif met_group(met) == "tenth":
            gm_2 = np.append(gm_2, rate)
        
        elif met_group(met) == "twentieth":
            gm_3 = np.append(gm_3, rate)
        
        elif met_group(met) == "fifty":
            gm_4 = np.append(gm_4, rate)
            
    
    #print(f"gM_1: {len(gM_1)} - {gM_1}")
    
    dp_fifth_max = np.sum(gM_1)/len(gM_1)
    dp_tenth_max = np.sum(gM_2)/len(gM_2)
    dp_twentieth_max = np.sum(gM_3)/len(gM_3)
    dp_fifty_max = np.sum(gM_4)/len(gM_4)
    
    dp_fifth_min = np.sum(gm_1)/len(gm_1)
    dp_tenth_min = np.sum(gm_2)/len(gm_2)
    dp_twentieth_min = np.sum(gm_3)/len(gm_3)
    dp_fifty_min = np.sum(gm_4)/len(gm_4)
    
    data_points_fifth_max_BNS = np.append(data_points_fifth_max_BNS, dp_fifth_max)
    data_points_tenth_max_BNS = np.append(data_points_tenth_max_BNS, dp_tenth_max)
    data_points_twentieth_max_BNS = np.append(data_points_twentieth_max_BNS, dp_twentieth_max)
    data_points_fifty_max_BNS = np.append(data_points_fifty_max_BNS, dp_fifty_max)

    data_points_fifth_min_BNS = np.append(data_points_fifth_min_BNS, dp_fifth_min)
    data_points_tenth_min_BNS = np.append(data_points_tenth_min_BNS, dp_tenth_min)
    data_points_twentieth_min_BNS = np.append(data_points_twentieth_min_BNS, dp_twentieth_min)
    data_points_fifty_min_BNS = np.append(data_points_fifty_min_BNS, dp_fifty_min)
    
    #print(f"Model {model} array sizes: {len(gM_1)} - {len(gM_2)} - {len(gM_3)} - {len(gM_4)} - {len(gm_1)} - {len(gm_2)} -{len(gm_3)} - {len(gm_4)}")
    
    if model == "F":
        print(f"Model F: {dp_fifth_min} - {dp_tenth_min} - {dp_twentieth_min} - {dp_fifty_min}")
        print(f"Model F: {dp_fifth_max} - {dp_tenth_max} - {dp_twentieth_max} - {dp_fifty_max}")
    #if model == "A" or model == "B":
       #print(f"Model {model}: {rel_rates_min}")

### TO GET BHNS DATA POINTS:

def Rates(metallicity_array, mask, weights_array, m_BH, m_NS, EoS, BH_spin): 
    
    Mbh = m_BH[mask]
    Mns = m_NS[mask]
    
    eos = EoS
    ens = ep.NS_EOS_properties(eos)
    spin_bh = BH_spin
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

        Mej_list = np.append(Mej_list, np.asarray(m_out))
    
    mask_final = Mej_list > 0
    
    metallicities_before = metallicity_array[mask]
    metallicities = np.log10(metallicities_before[mask_final])
    
    weights_before = weights_array[mask]
    weights = weights_before[mask_final]
    metallicity, frequency = np.unique(metallicities, return_counts=True)
    
    #print(f'The number of Candidates in Model {model_letter} is: {len(metallicities)}')
    
    weights_sum = []
    
    for n in np.asarray(metallicity):
        mask_metallicity = np.in1d(metallicities, np.asarray(n))
        weights_masked = weights[mask_metallicity]
        weights_sum = np.append(weights_sum, np.sum(weights_masked))
    
    rates = np.divide(weights_sum, 77598320)
    
    return rates, 10**metallicity

rates_1_max = []
rates_2_max = []
rates_3_max = []
rates_4_max = []

rates_1_min = []
rates_2_min = []
rates_3_min = []
rates_4_min = []

M_1 = []
M_2 = []
M_3 = []
M_4 = []

m_1 = []
m_2 = []
m_3 = []
m_4 = []

for ind, model in enumerate(list_of_models):
    
    metallicity = metallicity_data[ind]
    
    mask_max = mask_data_max[ind]
    
    mask_min = mask_data_min[ind]
    
    m_BH = mass_bh_data[ind]
    
    m_NS = mass_ns_data[ind]
    
    weights = weights_data[ind]
    
    rates_max_1, MET_1 = (Rates(metallicity_array=metallicity, mask=mask_max, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0.5))
    rates_min_1, met_1 = (Rates(metallicity_array=metallicity, mask=mask_min, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0.5))
    
    rates_1_max.append(rates_max_1*(10**0.7))
    rates_1_min.append(rates_min_1*(10**0.7))
    
    M_1.append(MET_1)
   
    
    m_1.append(met_1)
   

data_points_fifth_max_1 = []
data_points_tenth_max_1 = []
data_points_twentieth_max_1 = []
data_points_fifty_max_1 = []

data_points_fifth_min_1 = []
data_points_tenth_min_1 = []
data_points_twentieth_min_1 = []
data_points_fifty_min_1 = []

print("\nBHNS\n")

x1 = 0


z1 = 0

for ind, model in enumerate(list_of_models):
    
    A1 = []
    A2 = []
    A3 = []
    A4 = []

    a1 = []
    a2 = []
    a3 = []
    a4 = []

    for met, rate in zip(M_1[ind], rates_1_max[ind]):
        
        print(met)
        print(rate)
        
        if met_group(met) == "fifth":
            A1 = np.append(A1, rate)
        
        elif met_group(met) == "tenth":
            A2 = np.append(A2, rate)
        
        elif met_group(met) == "twentieth":
            A3 = np.append(A3, rate)
        
        elif met_group(met) == "fifty":
            A4 = np.append(A4, rate)
            
        
    for met, rate in zip(m_1[ind], rates_1_min[ind]):
        
        if met_group(met) == "fifth":
            a1 = np.append(a1, rate)
        
        elif met_group(met) == "tenth":
            a2 = np.append(a2, rate)
        
        elif met_group(met) == "twentieth":
            a3 = np.append(a3, rate)
        
        elif met_group(met) == "fifty":
            a4 = np.append(a4, rate)
    
    data_point_1_max_fifth = np.sum(A1)/len(A1)
    data_point_1_min_fifth = np.sum(a1)/len(a1)
    
    data_point_1_max_tenth = np.sum(A2)/len(A2)
    data_point_1_min_tenth = np.sum(a2)/len(a2)
    
    data_point_1_max_twentieth = np.sum(A3)/len(A3)
    data_point_1_min_twentieth = np.sum(a3)/len(a3)
    
    data_point_1_max_fifty = np.sum(A4)/len(A4)
    data_point_1_min_fifty = np.sum(a4)/len(a4)
    
    data_points_fifth_max_1 = np.append(data_points_fifth_max_1, data_point_1_max_fifth)
    data_points_tenth_max_1 = np.append(data_points_tenth_max_1, data_point_1_max_tenth)
    data_points_twentieth_max_1 = np.append(data_points_twentieth_max_1, data_point_1_max_twentieth)
    data_points_fifty_max_1 = np.append(data_points_fifty_max_1, data_point_1_max_fifty)
    
    data_points_fifth_min_1 = np.append(data_points_fifth_min_1, data_point_1_min_fifth)
    data_points_tenth_min_1 = np.append(data_points_tenth_min_1, data_point_1_min_tenth)
    data_points_twentieth_min_1 = np.append(data_points_twentieth_min_1, data_point_1_min_twentieth)
    data_points_fifty_min_1 = np.append(data_points_fifty_min_1, data_point_1_min_fifty)
    
    

ncols, nrows = 1,1
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(26,8), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharex=True)
f.subplots_adjust(hspace=0)

axe.axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='lightgrey', alpha=0.5, lw=0, label = r'$\rm{Reticulum \: II \: required \: rates}$')
axe.axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='dimgrey', alpha=0.6, lw=0, label = r'$\rm{Tucana \: III \: required \: rates}$')

### FIRST PLOT:

axe.scatter(x_ticks_1, data_points_fifth_max_1, c = 'darkblue', marker = 'o', s=200)
axe.scatter(x_ticks_1, data_points_fifth_min_1, marker = 'o', facecolors = 'none', edgecolors='darkblue', s=200)

axe.scatter(x_ticks_1, data_points_tenth_max_1, c = 'green', marker = 'o',s=200)
axe.scatter(x_ticks_1, data_points_tenth_min_1, marker = 'o', facecolors = 'none', edgecolors='green', s=200)

axe.scatter(x_ticks_1, data_points_twentieth_max_1, c = 'limegreen', marker = 'o', s=200)
axe.scatter(x_ticks_1, data_points_twentieth_min_1, marker = 'o', facecolors = 'none', edgecolors='limegreen',s=200)

axe.scatter(x_ticks_1, data_points_fifty_max_1, c = 'yellow', marker = 'o', s=200)
axe.scatter(x_ticks_1, data_points_fifty_min_1, marker = 'o', facecolors = 'none', edgecolors='yellow',s=200)

axe.scatter(x_ticks_2, data_points_fifth_max_BNS, c = 'darkblue', marker = 'D', s=200)
axe.scatter(x_ticks_2, data_points_fifth_min_BNS, marker = 'D', facecolors = 'none', edgecolors='darkblue',s=200)

axe.scatter(x_ticks_2, data_points_tenth_max_BNS, c = 'green', marker = 'D', s=200)
axe.scatter(x_ticks_2, data_points_tenth_min_BNS, marker = 'D', facecolors = 'none', edgecolors='green',s=200)

axe.scatter(x_ticks_2, data_points_twentieth_max_BNS, c = 'limegreen', marker = 'D', s=200)
axe.scatter(x_ticks_2, data_points_twentieth_min_BNS,  marker = 'D', facecolors = 'none', edgecolors='limegreen',s=200)

axe.scatter(x_ticks_2, data_points_fifty_max_BNS, c = 'yellow', marker = 'D', s=200)
axe.scatter(x_ticks_2, data_points_fifty_min_BNS,  marker = 'D', facecolors = 'none', edgecolors='yellow',s=200)


axe.set_ylim((10e-9, 10e-5))


axe.set_yscale('log')


layoutAxesNoXlabel(ax=axe, nameY=nameY, fontsize=30, setMinor=False)


special_x_labels = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'D', 'T']

#axe[3].set_xticks(ticks=x_ticks, labels=special_x_labels, fontsize=15, rotation=30)
#axe[3].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

axe.set_xticks(ticks=x_ticks, labels=xlabels, fontsize=20, rotation=30)
#axe[3].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=15, rotation=30)


axe.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

axe.grid()

axe.yaxis.set_label_coords(-0.07, 0.5)


#### Label's stuff:

plt.rcParams['text.usetex'] = True

#nameY = r'$\mathcal{Z}_{\rm{\odot}}  \ \  [\rm{M}_{\odot}^{-1}]\   $'

axe.scatter([20,21], [10e-10,10e-10], marker = 'o', c='black', label = r'$ \rm{BHNS} + 10^{9}  \rm{M}_{\odot} \ UFD$' )
axe.scatter([22,23], [10e-10,10e-10], marker = 'o', facecolors='none', edgecolors='black', label = r'$ \rm{BHNS} + 10^{8}  \rm{M}_{\odot} \ UFD$')

axe.scatter([24,25], [10e-10,10e-10], marker = 'D', c='black', label = r'$ \rm{BNS} + 10^{9}  \rm{M}_{\odot} \ UFD$')
axe.scatter([26,27], [10e-10,10e-10], marker = 'D', facecolors='none', edgecolors='black', label = r'$ \rm{BNS} + 10^{8}  \rm{M}_{\odot} \ UFD$')

axe.plot([10,11], [10e-10,10e-10], c='darkblue', label = r'${Z}_{\odot}/10 - {Z}_{\odot}/5$')
axe.plot([12,13], [10e-10,10e-10], c='green', label = r'${Z}_{\odot}/20 - {Z}_{\odot}/10$')
axe.plot([14,15], [10e-10,10e-10], c='limegreen', label = r'${Z}_{\odot}/50- {Z}_{\odot}/20$')
axe.plot([16,17], [10e-10,10e-10], c='yellow', label = r'$0.0001 - {Z}_{\odot}/50$')

handles, labels = axe.get_legend_handles_labels()

#axe.legend(ncols=4)

####

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.95)

axe.annotate(r'%s'%('Stiff DD2  + 0.5 BH spin'), xy=(0.03, .967), xycoords='axes fraction', fontsize = 18, weight = 'bold', ha='left', va="top",bbox=bbox_props)

#f.legend(handles, labels , fontsize='28', ncol=6, markerscale=2.5, loc='outside upper center')

plt.tight_layout()

lgd2 = plt.legend(handles, labels , fontsize=24, ncol=5,\
         bbox_to_anchor=(0.98, 1.48), markerscale=2)

plt.savefig('./RatesPlot_DD2_test_3.pdf', bbox_extra_artists=(lgd2,), bbox_inches='tight', dpi = 600)

plt.show()
plt.close()