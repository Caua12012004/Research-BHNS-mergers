from IPython.core.display import display, HTML
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

x_ticks_1 = [9.5, 19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5, 99.5, 109.5, 119.5, 129.5, 139.5, 149.5, 159.5, 169.5, 179.5, 189.5]

x_ticks_2 = [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5, 100.5, 110.5, 120.5, 130.5, 140.5, 150.5, 160.5, 170.5, 180.5, 190.5]

xlabels = [dic_xlabels.get(t, x_ticks[i]) for i,t in enumerate(x_ticks)]

### TO GET BNS DATA POINTS:

data_points_fifth_max_BNS = []
data_points_tenth_max_BNS = []
data_points_twentieth_max_BNS = []
data_points_fifty_max_BNS = []

data_points_fifth_min_BNS = []
data_points_tenth_min_BNS = []
data_points_twentieth_min_BNS = []
data_points_fifty_min_BNS = []

for ind, model in enumerate(list_of_models):
    
    metallicity = metallicity_data_BNS[ind]
    
    #25:31 = (metallicity <= Z_solar/5) & (metallicity > Z_solar/10)
    #18:25 = (metallicity <= Z_solar/10) & (metallicity > Z_solar/20)
    #10:18 = (metallicity <= Z_solar/20) & (metallicity > Z_solar/50)
    #0:10 = (metallicity <= Z_solar/50)
    
    rel_rates_max = np.asarray(model_rates_max_BNS[ind])
    rel_rates_min = np.asarray(model_rates_min_BNS[ind])
    
    dp_fifth_max = np.sum(rel_rates_max[25:31])/len(rel_rates_max[25:31])
    dp_tenth_max = np.sum(rel_rates_max[18:25])/len(rel_rates_max[18:25])
    dp_twentieth_max = np.sum(rel_rates_max[10:18])/len(rel_rates_max[10:18])
    dp_fifty_max = np.sum(rel_rates_max[0:10])/len(rel_rates_max[0:10])
    
    dp_fifth_min = np.sum(rel_rates_min[25:31])/len(rel_rates_min[25:31])
    dp_tenth_min = np.sum(rel_rates_min[18:25])/len(rel_rates_min[18:25])
    dp_twentieth_min = np.sum(rel_rates_min[10:18])/len(rel_rates_min[10:18])
    dp_fifty_min = np.sum(rel_rates_min[0:10])/len(rel_rates_min[0:10])
    
    data_points_fifth_max_BNS = np.append(data_points_fifth_max_BNS, dp_fifth_max)
    data_points_tenth_max_BNS = np.append(data_points_tenth_max_BNS, dp_tenth_max)
    data_points_twentieth_max_BNS = np.append(data_points_twentieth_max_BNS, dp_twentieth_max)
    data_points_fifty_max_BNS = np.append(data_points_fifty_max_BNS, dp_fifty_max)

    data_points_fifth_min_BNS = np.append(data_points_fifth_min_BNS, dp_fifth_min)
    data_points_tenth_min_BNS = np.append(data_points_tenth_min_BNS, dp_tenth_min)
    data_points_twentieth_min_BNS = np.append(data_points_twentieth_min_BNS, dp_twentieth_min)
    data_points_fifty_min_BNS = np.append(data_points_fifty_min_BNS, dp_fifty_min)
    

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
    
    return rates

rates_1_max = []
rates_2_max = []
rates_3_max = []
rates_4_max = []

rates_1_min = []
rates_2_min = []
rates_3_min = []
rates_4_min = []

lengths_1_max = np.asarray([0])
lengths_1_min = np.asarray([0])

lengths_2_max = np.asarray([0])
lengths_2_min = np.asarray([0])

lengths_3_max = np.asarray([0])
lengths_3_min = np.asarray([0])

lengths_4_max = np.asarray([0])
lengths_4_min = np.asarray([0])

for ind, model in enumerate(list_of_models):
    
    metallicity = metallicity_data[ind]
    
    mask_max = mask_data_max[ind]
    
    mask_min = mask_data_min[ind]
    
    m_BH = mass_bh_data[ind]
    
    m_NS = mass_ns_data[ind]
    
    weights = weights_data[ind]
    
    rates_max_1 = (Rates(metallicity_array=metallicity, mask=mask_max, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0.5))*(10**0.7)
    rates_min_1 = (Rates(metallicity_array=metallicity, mask=mask_min, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0.5))*(10**0.7)
    
    rates_max_2 = (Rates(metallicity_array=metallicity, mask=mask_max, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0))*(10**0.7)
    rates_min_2 = (Rates(metallicity_array=metallicity, mask=mask_min, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='DD2', BH_spin=0))*(10**0.7)
    
    rates_max_3 = (Rates(metallicity_array=metallicity, mask=mask_max, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='SFHo', BH_spin=0.5))*(10**0.7)
    rates_min_3 = (Rates(metallicity_array=metallicity, mask=mask_min, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='SFHo', BH_spin=0.5))*(10**0.7)
    
    try:
        
        rates_max_4 = (Rates(metallicity_array=metallicity, mask=mask_max, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='SFHo', BH_spin=0))*(10**0.7)
        rates_min_4 = (Rates(metallicity_array=metallicity, mask=mask_min, weights_array=weights, m_BH=m_BH, m_NS=m_NS, EoS='SFHo', BH_spin=0))*(10**0.7)
    
    except ValueError:
        
        rates_max_4 = np.asarray([0])
        rates_min_4 = np.asarray([0])
    
    rates_1_max = np.append(rates_1_max, rates_max_1)
    rates_1_min = np.append(rates_1_min, rates_min_1)
    
    rates_2_max = np.append(rates_2_max, rates_max_2)
    rates_2_min = np.append(rates_2_min, rates_min_2)
    
    rates_3_max = np.append(rates_3_max, rates_max_3)
    rates_3_min = np.append(rates_3_min, rates_min_3)
    
    rates_4_max = np.append(rates_4_max, rates_max_4)
    rates_4_min = np.append(rates_4_min, rates_min_4)
    
    """lengths_1_max = np.append(lengths_1_max, len(rates_1_max))
    lengths_1_min = np.append(lengths_1_min, len(rates_1_min))
    
    lengths_2_max = np.append(lengths_2_max, len(rates_2_max))
    lengths_2_min = np.append(lengths_2_min, len(rates_2_min))
    
    lengths_3_max = np.append(lengths_3_max, len(rates_3_max))
    lengths_3_min = np.append(lengths_3_min, len(rates_3_min))
    
    lengths_4_max = np.append(lengths_4_max, len(rates_4_max))
    lengths_4_min = np.append(lengths_4_min, len(rates_4_min))"""

data_points_fifth_max_1 = []
data_points_tenth_max_1 = []
data_points_twentieth_max_1 = []
data_points_fifty_max_1 = []

data_points_fifth_min_1 = []
data_points_tenth_min_1 = []
data_points_twentieth_min_1 = []
data_points_fifty_min_1 = []

data_points_fifth_max_2 = []
data_points_tenth_max_2 = []
data_points_twentieth_max_2 = []
data_points_fifty_max_2 = []

data_points_fifth_min_2 = []
data_points_tenth_min_2 = []
data_points_twentieth_min_2 = []
data_points_fifty_min_2 = []

data_points_fifth_max_3 = []
data_points_tenth_max_3 = []
data_points_twentieth_max_3 = []
data_points_fifty_max_3 = []

data_points_fifth_min_3 = []
data_points_tenth_min_3 = []
data_points_twentieth_min_3 = []
data_points_fifty_min_3 = []

data_points_fifth_max_4 = []
data_points_tenth_max_4 = []
data_points_twentieth_max_4 = []
data_points_fifty_max_4 = []

data_points_fifth_min_4 = []
data_points_tenth_min_4 = []
data_points_twentieth_min_4 = []
data_points_fifty_min_4 = []

"""base_lenght_1_max = 0
base_lenght_2_max = 0
base_lenght_3_max = 0
base_lenght_4_max = 0

base_lenght_1_min = 0
base_lenght_2_min = 0
base_lenght_3_min = 0
base_lenght_4_min = 0"""

base_length = 0
length = 31

for ind, model in enumerate(list_of_models):
    
    rel_metallicity = metallicity_data[ind]
    
    #25:31 = (rel_metallicity <= Z_solar/5) & (rel_metallicity > Z_solar/10)
    #18:25 = (rel_metallicity <= Z_solar/10) & (rel_metallicity > Z_solar/20)
    #10:18 = (rel_metallicity <= Z_solar/20) & (rel_metallicity > Z_solar/50)
    #0:10 = (rel_metallicity <= Z_solar/50)
    
    """length_1_max = lengths_1_max[ind]
    length_1_min = lengths_1_min[ind]
    
    length_2_max = lengths_2_max[ind]
    length_2_min = lengths_2_min[ind]
    
    length_3_max = lengths_3_max[ind]
    length_3_min = lengths_3_min[ind]
    
    length_4_max = lengths_4_max[ind]
    length_4_min = lengths_4_min[ind]"""
    
    ### DATA POINTS 1:
    
    data_point_1_max_fifth = np.sum(rates_1_max[base_length:base_length+length][25:31])/len(rates_1_max[base_length:base_length+length][25:31])
    data_point_1_min_fifth = np.sum(rates_1_min[base_length:base_length+length][25:31])/len(rates_1_max[base_length:base_length+length][25:31])
    
    data_point_1_max_tenth = np.sum(rates_1_max[base_length:base_length+length][18:25])/len(rates_1_max[base_length:base_length+length][18:25])
    data_point_1_min_tenth = np.sum(rates_1_min[base_length:base_length+length][18:25])/len(rates_1_max[base_length:base_length+length][18:25])
    
    data_point_1_max_twentieth = np.sum(rates_1_max[base_length:base_length+length][10:18])/len(rates_1_max[base_length:base_length+length][10:18])
    data_point_1_min_twentieth = np.sum(rates_1_min[base_length:base_length+length][10:18])/len(rates_1_max[base_length:base_length+length][10:18])
    
    data_point_1_max_fifty = np.sum(rates_1_max[base_length:base_length+length][0:10])/len(rates_1_max[base_length:base_length+length][0:10])
    data_point_1_min_fifty = np.sum(rates_1_min[base_length:base_length+length][0:10])/len(rates_1_max[base_length:base_length+length][0:10])
    
    data_points_fifth_max_1 = np.append(data_points_fifth_max_1, data_point_1_max_fifth)
    data_points_tenth_max_1 = np.append(data_points_tenth_max_1, data_point_1_max_tenth)
    data_points_twentieth_max_1 = np.append(data_points_twentieth_max_1, data_point_1_max_twentieth)
    data_points_fifty_max_1 = np.append(data_points_fifty_max_1, data_point_1_max_fifty)
    
    data_points_fifth_min_1 = np.append(data_points_fifth_min_1, data_point_1_min_fifth)
    data_points_tenth_min_1 = np.append(data_points_tenth_min_1, data_point_1_min_tenth)
    data_points_twentieth_min_1 = np.append(data_points_twentieth_min_1, data_point_1_min_twentieth)
    data_points_fifty_min_1 = np.append(data_points_fifty_min_1, data_point_1_min_fifty)
    
    #base_lenght_1_max = length_1_max
    #base_lenght_1_min = length_1_min

    ### DATA POINTS 2:
    
    data_point_2_max_fifth = np.sum(rates_2_max[base_length:base_length+length][25:31])/len(rates_2_max[base_length:base_length+length][25:31])
    data_point_2_min_fifth = np.sum(rates_2_min[base_length:base_length+length][25:31])/len(rates_2_min[base_length:base_length+length][25:31])
    
    data_point_2_max_tenth = np.sum(rates_2_max[base_length:base_length+length][18:25])/len(rates_2_max[base_length:base_length+length][18:25])
    data_point_2_min_tenth = np.sum(rates_2_min[base_length:base_length+length][18:25])/len(rates_2_min[base_length:base_length+length][18:25])
    
    data_point_2_max_twentieth = np.sum(rates_2_max[base_length:base_length+length][10:18])/len(rates_2_max[base_length:base_length+length][10:18])
    data_point_2_min_twentieth = np.sum(rates_2_min[base_length:base_length+length][10:18])/len(rates_2_min[base_length:base_length+length][10:18])
    
    data_point_2_max_fifty = np.sum(rates_2_max[base_length:base_length+length][0:10])/len(rates_2_max[base_length:base_length+length][0:10])
    data_point_2_min_fifty = np.sum(rates_2_min[base_length:base_length+length][0:10])/len(rates_2_min[base_length:base_length+length][0:10])
    
    data_points_fifth_max_2 = np.append(data_points_fifth_max_2, data_point_2_max_fifth)
    data_points_tenth_max_2 = np.append(data_points_tenth_max_2, data_point_2_max_tenth)
    data_points_twentieth_max_2 = np.append(data_points_twentieth_max_2, data_point_2_max_twentieth)
    data_points_fifty_max_2 = np.append(data_points_fifty_max_2, data_point_2_max_fifty)
    
    data_points_fifth_min_2 = np.append(data_points_fifth_min_2, data_point_2_min_fifth)
    data_points_tenth_min_2 = np.append(data_points_tenth_min_2, data_point_2_min_tenth)
    data_points_twentieth_min_2 = np.append(data_points_twentieth_min_2, data_point_2_min_twentieth)
    data_points_fifty_min_2 = np.append(data_points_fifty_min_2, data_point_2_min_fifty)
    
    #base_lenght_2_max = length_2_max
    #base_lenght_2_min = length_2_min
    
    ### DATA POINTS 3:
    
    data_point_3_max_fifth = np.sum(rates_3_max[base_length:base_length+length][25:31])/len(rates_3_max[base_length:base_length+length][25:31])
    data_point_3_min_fifth = np.sum(rates_3_min[base_length:base_length+length][25:31])/len(rates_3_max[base_length:base_length+length][25:31])
    
    data_point_3_max_tenth = np.sum(rates_3_max[base_length:base_length+length][18:25])/len(rates_3_max[base_length:base_length+length][18:25])
    data_point_3_min_tenth = np.sum(rates_3_min[base_length:base_length+length][18:25])/len(rates_3_max[base_length:base_length+length][18:25])
    
    data_point_3_max_twentieth = np.sum(rates_3_max[base_length:base_length+length][10:18])/len(rates_3_max[base_length:base_length+length][10:18])
    data_point_3_min_twentieth = np.sum(rates_3_min[base_length:base_length+length][10:18])/len(rates_3_max[base_length:base_length+length][10:18])
    
    data_point_3_max_fifty = np.sum(rates_3_max[base_length:base_length+length][0:10])/len(rates_3_max[base_length:base_length+length][0:10])
    data_point_3_min_fifty = np.sum(rates_3_min[base_length:base_length+length][0:10])/len(rates_3_max[base_length:base_length+length][0:10])
    
    data_points_fifth_max_3 = np.append(data_points_fifth_max_3, data_point_3_max_fifth)
    data_points_tenth_max_3 = np.append(data_points_tenth_max_3, data_point_3_max_tenth)
    data_points_twentieth_max_3 = np.append(data_points_twentieth_max_3, data_point_3_max_twentieth)
    data_points_fifty_max_3 = np.append(data_points_fifty_max_3, data_point_3_max_fifty)
    
    data_points_fifth_min_3 = np.append(data_points_fifth_min_3, data_point_3_min_fifth)
    data_points_tenth_min_3 = np.append(data_points_tenth_min_3, data_point_3_min_tenth)
    data_points_twentieth_min_3 = np.append(data_points_twentieth_min_3, data_point_3_min_twentieth)
    data_points_fifty_min_3 = np.append(data_points_fifty_min_3, data_point_3_min_fifty)
    
    #base_lenght_3_max = length_3_max
    #base_lenght_3_min = length_3_min
    
    ### DATA POINTS 4:
    
    data_point_4_max_fifth = np.sum(rates_4_max[base_length:base_length+length][25:31])/len(rates_4_max[base_length:base_length+length][25:31])
    data_point_4_min_fifth = np.sum(rates_4_min[base_length:base_length+length][25:31])/len(rates_4_max[base_length:base_length+length][25:31])
    
    data_point_4_max_tenth = np.sum(rates_4_max[base_length:base_length+length][18:25])/len(rates_4_max[base_length:base_length+length][18:25])
    data_point_4_min_tenth = np.sum(rates_4_min[base_length:base_length+length][18:25])/len(rates_4_max[base_length:base_length+length][18:25])
    
    data_point_4_max_twentieth = np.sum(rates_4_max[base_length:base_length+length][10:18])/len(rates_4_max[base_length:base_length+length][10:18])
    data_point_4_min_twentieth = np.sum(rates_4_min[base_length:base_length+length][10:18])/len(rates_4_max[base_length:base_length+length][10:18])
    
    data_point_4_max_fifty = np.sum(rates_4_max[base_length:base_length+length][0:10])/len(rates_4_max[base_length:base_length+length][0:10])
    data_point_4_min_fifty = np.sum(rates_4_min[base_length:base_length+length][0:10])/len(rates_4_max[base_length:base_length+length][0:10])
    
    data_points_fifth_max_4 = np.append(data_points_fifth_max_4, data_point_4_max_fifth)
    data_points_tenth_max_4 = np.append(data_points_tenth_max_4, data_point_4_max_tenth)
    data_points_twentieth_max_4 = np.append(data_points_twentieth_max_4, data_point_4_max_twentieth)
    data_points_fifty_max_4 = np.append(data_points_fifty_max_4, data_point_4_max_fifty)
    
    data_points_fifth_min_4 = np.append(data_points_fifth_min_4, data_point_4_min_fifth)
    data_points_tenth_min_4 = np.append(data_points_tenth_min_4, data_point_4_min_tenth)
    data_points_twentieth_min_4 = np.append(data_points_twentieth_min_4, data_point_4_min_twentieth)
    data_points_fifty_min_4 = np.append(data_points_fifty_min_4, data_point_4_min_fifty)
    
    #base_lenght_4_max = length_4_max
    #base_lenght_4_min = length_4_min
    
    base_length = base_length + 31
    length = length + 31
    

ncols, nrows = 1,4
f, axe= plt.subplots(ncols=ncols,nrows=nrows,figsize=(24,30), gridspec_kw={"width_ratios":1*np.ones(ncols), "height_ratios":1*np.ones(nrows)}, sharex=True)
f.subplots_adjust(hspace=0)

axe[0].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe[0].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

axe[1].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe[1].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

axe[2].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0)
axe[2].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0)

axe[3].axhspan(1/Max_Reticulum_II_Mass, 1/Min_Reticulum_II_Mass, color='grey', alpha=0.5, lw=0, label='Reticulum II required rates')
axe[3].axhspan(1/Max_Tucana_III_Mass, 1/Min_Tucana_III_Mass, color='teal', alpha=0.6, lw=0, label='Tucana III required rates')

### FIRST PLOT:

axe[0].scatter(x_ticks_1, data_points_fifth_max_1, c = 'teal', marker = 'o', s=36)
axe[0].scatter(x_ticks_1, data_points_fifth_min_1, c = 'mediumturquoise', marker = 'o', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_1, data_points_tenth_max_1, c = 'teal', marker = '^', s=36)
axe[0].scatter(x_ticks_1, data_points_tenth_min_1, c = 'mediumturquoise', marker = '^', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_1, data_points_twentieth_max_1, c = 'teal', marker = 's', s=36)
axe[0].scatter(x_ticks_1, data_points_twentieth_min_1, c = 'mediumturquoise', marker = 's', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_1, data_points_fifty_max_1, c = 'teal', marker = 'D', s=36)
axe[0].scatter(x_ticks_1, data_points_fifty_min_1, c = 'mediumturquoise', marker = 'D', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_2, data_points_fifth_max_BNS, c = 'goldenrod', marker = 'o', s=36)
axe[0].scatter(x_ticks_2, data_points_fifth_min_BNS, c = 'gold', marker = 'o', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_2, data_points_tenth_max_BNS, c = 'goldenrod', marker = '^', s=36)
axe[0].scatter(x_ticks_2, data_points_tenth_min_BNS, c = 'gold', marker = '^', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_2, data_points_twentieth_max_BNS, c = 'goldenrod', marker = 's', s=36)
axe[0].scatter(x_ticks_2, data_points_twentieth_min_BNS, c = 'gold', marker = 's', facecolors = 'none', s=36)

axe[0].scatter(x_ticks_2, data_points_fifty_max_BNS, c = 'goldenrod', marker = 'D', s=36)
axe[0].scatter(x_ticks_2, data_points_fifty_min_BNS, c = 'gold', marker = 'D', facecolors = 'none', s=36)

### SECOND PLOT:

axe[1].scatter(x_ticks_1, data_points_fifth_max_2, c = 'teal', marker = 'o', s=36)
axe[1].scatter(x_ticks_1, data_points_fifth_min_2, c = 'mediumturquoise', marker = 'o', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_1, data_points_tenth_max_2, c = 'teal', marker = '^', s=36)
axe[1].scatter(x_ticks_1, data_points_tenth_min_2, c = 'mediumturquoise', marker = '^', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_1, data_points_twentieth_max_2, c = 'teal', marker = 's', s=36)
axe[1].scatter(x_ticks_1, data_points_twentieth_min_2, c = 'mediumturquoise', marker = 's', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_1, data_points_fifty_max_2, c = 'teal', marker = 'D', s=36)
axe[1].scatter(x_ticks_1, data_points_fifty_min_2, c = 'mediumturquoise', marker = 'D', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_2, data_points_fifth_max_BNS, c = 'goldenrod', marker = 'o', s=36)
axe[1].scatter(x_ticks_2, data_points_fifth_min_BNS, c = 'gold', marker = 'o', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_2, data_points_tenth_max_BNS, c = 'goldenrod', marker = '^', s=36)
axe[1].scatter(x_ticks_2, data_points_tenth_min_BNS, c = 'gold', marker = '^', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_2, data_points_twentieth_max_BNS, c = 'goldenrod', marker = 's', s=36)
axe[1].scatter(x_ticks_2, data_points_twentieth_min_BNS, c = 'gold', marker = 's', facecolors = 'none', s=36)

axe[1].scatter(x_ticks_2, data_points_fifty_max_BNS, c = 'goldenrod', marker = 'D', s=36)
axe[1].scatter(x_ticks_2, data_points_fifty_min_BNS, c = 'gold', marker = 'D', facecolors = 'none', s=36)

### THIRD PLOT:

axe[2].scatter(x_ticks_1, data_points_fifth_max_3, c = 'teal', marker = 'o', s=36)
axe[2].scatter(x_ticks_1, data_points_fifth_min_3, c = 'mediumturquoise', marker = 'o', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_1, data_points_tenth_max_3, c = 'teal', marker = '^', s=36)
axe[2].scatter(x_ticks_1, data_points_tenth_min_3, c = 'mediumturquoise', marker = '^', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_1, data_points_twentieth_max_3, c = 'teal', marker = 's', s=36)
axe[2].scatter(x_ticks_1, data_points_twentieth_min_3, c = 'mediumturquoise', marker = 's', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_1, data_points_fifty_max_3, c = 'teal', marker = 'D', s=36)
axe[2].scatter(x_ticks_1, data_points_fifty_min_3, c = 'mediumturquoise', marker = 'D', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_2, data_points_fifth_max_BNS, c = 'goldenrod', marker = 'o', s=36)
axe[2].scatter(x_ticks_2, data_points_fifth_min_BNS, c = 'gold', marker = 'o', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_2, data_points_tenth_max_BNS, c = 'goldenrod', marker = '^', s=36)
axe[2].scatter(x_ticks_2, data_points_tenth_min_BNS, c = 'gold', marker = '^', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_2, data_points_twentieth_max_BNS, c = 'goldenrod', marker = 's', s=36)
axe[2].scatter(x_ticks_2, data_points_twentieth_min_BNS, c = 'gold', marker = 's', facecolors = 'none', s=36)

axe[2].scatter(x_ticks_2, data_points_fifty_max_BNS, c = 'goldenrod', marker = 'D', s=36)
axe[2].scatter(x_ticks_2, data_points_fifty_min_BNS, c = 'gold', marker = 'D', facecolors = 'none', s=36)

### FOURTH PLOT:

axe[3].scatter(x_ticks_1, data_points_fifth_max_4, c = 'teal', marker = 'o', s=36)
axe[3].scatter(x_ticks_1, data_points_fifth_min_4, c = 'mediumturquoise', marker = 'o', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_1, data_points_tenth_max_4, c = 'teal', marker = '^', s=36)
axe[3].scatter(x_ticks_1, data_points_tenth_min_4, c = 'mediumturquoise', marker = '^', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_1, data_points_twentieth_max_4, c = 'teal', marker = 's', s=36)
axe[3].scatter(x_ticks_1, data_points_twentieth_min_4, c = 'mediumturquoise', marker = 's', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_1, data_points_fifty_max_4, c = 'teal', marker = 'D', s=36)
axe[3].scatter(x_ticks_1, data_points_fifty_min_4, c = 'mediumturquoise', marker = 'D', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_2, data_points_fifth_max_BNS, c = 'goldenrod', marker = 'o', s=36)
axe[3].scatter(x_ticks_2, data_points_fifth_min_BNS, c = 'gold', marker = 'o', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_2, data_points_tenth_max_BNS, c = 'goldenrod', marker = '^', s=36)
axe[3].scatter(x_ticks_2, data_points_tenth_min_BNS, c = 'gold', marker = '^', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_2, data_points_twentieth_max_BNS, c = 'goldenrod', marker = 's', s=36)
axe[3].scatter(x_ticks_2, data_points_twentieth_min_BNS, c = 'gold', marker = 's', facecolors = 'none', s=36)

axe[3].scatter(x_ticks_2, data_points_fifty_max_BNS, c = 'goldenrod', marker = 'D', s=36)
axe[3].scatter(x_ticks_2, data_points_fifty_min_BNS, c = 'gold', marker = 'D', facecolors = 'none', s=36)


axe[0].set_ylim((10e-9, 10e-5))
axe[1].set_ylim((10e-9, 10e-5))
axe[2].set_ylim((10e-9, 10e-5))
axe[3].set_ylim((10e-9, 10e-5))

axe[0].set_yscale('log')
axe[1].set_yscale('log')
axe[2].set_yscale('log')
axe[3].set_yscale('log')

layoutAxesNoXlabel(ax=axe[0], nameY=nameY, fontsize=24, setMinor=False)
layoutAxesNoXlabel(ax=axe[1], nameY=nameY, fontsize=24, setMinor=False)
layoutAxesNoXlabel(ax=axe[2], nameY=nameY, fontsize=24, setMinor=False)
layoutAxesNoXlabel(ax=axe[3], nameY=nameY, fontsize=24, setMinor=False)

special_x_labels = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

axe[3].set_xticks(ticks=x_ticks, labels=special_x_labels, fontsize=15, rotation=30)
axe[3].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

axe[0].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=15, rotation=30)
axe[1].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=15, rotation=30)
axe[2].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=15, rotation=30)
#axe[3].set_xticks(ticks=x_ticks, labels=xlabels, fontsize=15, rotation=30)


axe[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axe[1].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
axe[2].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)
#axe[3].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False)

#axe[0].legend(loc='upper left', fontsize='x-large', ncols=2)
#axe[1].legend(loc='upper left', fontsize='x-large', ncols=2)
#axe[2].legend(loc='upper left', fontsize='x-large', ncols=2)
#axe[3].legend(loc='upper left', fontsize='x-large', ncols=2)

axe[0].grid()
axe[1].grid()
axe[2].grid()
axe[3].grid()

"""axe[0].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=16)
axe[0].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

axe[1].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=16)
axe[1].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

axe[2].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=16)
axe[2].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)

axe[3].text(3, 1.05/Min_Tucana_III_Mass, 'Tucana III required rates' , fontsize=16)
axe[3].text(3, 1.032/Max_Reticulum_II_Mass, 'Reticulum II required rates' , fontsize=14)"""

#axe[0].text(170, (1.032/Max_Reticulum_II_Mass)*1.4, 'Stiff DD2 + 0.5 BH spin' , fontsize=16, fontweight='bold')
#axe[1].text(170, (1.032/Max_Reticulum_II_Mass)*1.4, 'Stiff DD2 + 0 BH spin' , fontsize=16, fontweight='bold')
#axe[2].text(170, (1.032/Max_Reticulum_II_Mass)*1.4, 'Soft SFHo + 0.5 spin' , fontsize=16, fontweight='bold')
#axe[3].text(170, (1.032/Max_Reticulum_II_Mass)*1.4, 'Soft SFHo + 0 spin' , fontsize=16, fontweight='bold')

axe[0].yaxis.set_label_coords(-0.07, 0.5)
axe[1].yaxis.set_label_coords(-0.07, 0.5)
axe[2].yaxis.set_label_coords(-0.07, 0.5)
axe[3].yaxis.set_label_coords(-0.07, 0.5)

handles, labels = axe[3].get_legend_handles_labels()

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.95)

axe[0].annotate(r'%s'%('Stiff DD2 + 0.5 BH spin'), xy=(0.03, .967), xycoords='axes fraction', fontsize = 18, weight = 'bold', ha='left', va="top",bbox=bbox_props)
axe[1].annotate(r'%s'%('Stiff DD2 + 0 BH spin'), xy=(0.03, .967), xycoords='axes fraction', fontsize = 18, weight = 'bold', ha='left', va="top",bbox=bbox_props)
axe[2].annotate(r'%s'%('Soft SFHo + 0.5 spin'), xy=(0.03, .967), xycoords='axes fraction', fontsize = 18, weight = 'bold', ha='left', va="top",bbox=bbox_props)
axe[3].annotate(r'%s'%('Soft SFHo + 0 spin'), xy=(0.03, .967), xycoords='axes fraction', fontsize = 18, weight = 'bold', ha='left', va="top",bbox=bbox_props)

plt.tight_layout()

lgd2=plt.legend(handles, labels , fontsize='28', ncol=4, bbox_to_anchor=(0.98, 4.5))

plt.savefig('./RatesPlot_test_2.png', bbox_inches='tight', dpi = 600)

plt.show()
plt.close()
        