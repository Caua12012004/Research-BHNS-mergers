import sys
import numpy as np  
import BH_properties as BH_prop  

  
  
def M_out_f18_lambda(mbh,mns,x,lamb):
    """
    Mass M_out [Msun] of the material that remains outside the BH after BHNS merger. New fit by [Foucart et al. 18].
    Function of lambda parameter of NS (dimensionless quadrupolar tidal deformability).
    Input: 
         mbh = BH mass [Msun]
         mns = NS mass [Msun]
         x = BH spin parameter [-]
         lamb = lambda parameter of NS [-]		
    """
    alpha = 0.308
    beta = 0.124
    gamma = 0.283
    delta = 1.536

    rho = (15.*lamb)**(-1./5.)
    Q = mbh/mns
    eta = Q/((1.+Q)**2)

    comp = compactness(lamb)
    mnsb = m_to_mb(mns,comp)
    
    mfit = np.maximum(0., alpha * (1. - 2.*rho) / (eta**(1./3.))  - beta * BH_prop.R_isco(x,mbh,'no') * rho / eta + gamma)
    return (mfit**delta)*mnsb

    #return np.maximum((alpha*(1.-2.*rho)/np.power(eta,1./3.)-(beta*BH_prop.R_isco(x,mbh,'no')*rho/eta)+gamma)*mnsb,0.)**delta

		
def Mej_Kawaguchi16_lambda(MBH,MNS,chi,i_tilt,lamb):
    """
    Mass M_ej [Msun] of ejecta. 
    Function of lambda parameter of NS (dimensionless quadrupolar tidal deformability).
    Input:
        MBH    : BH mass [Msun]
        MNS    : NS mass [Msun]
        chi    : BH dimensionless spin parameter [-]
        i_tilt : tilt angle between BH spin and orbit plane [rad] 
        lamb = lambda parameter of NS [-]	
    """
    #Kawaguchi constants
    a1 = 4.464e-2
    a2 = 2.269e-3
    a3 = 2.431
    a4 = -0.4159
    n1 = 0.2497
    n2 = 1.352
    Q = MBH/MNS
    
    C = compactness(lamb)
    MNSb = m_to_mb(MNS,C)   
        
    risco = BH_prop.R_isco(chi*np.cos(i_tilt),MBH,'no')
    KW = a1*(Q**n1)*(1.-2.*C)/C - a2*(Q**n2)*risco + a3*(1.-MNS/MNSb) + a4
    if np.isscalar(KW):
        if KW>1.:
            KW = 1.
        elif KW<0.:
            KW = 0.
    else:
        KW[KW>1.]=1.
        KW[KW<0.]=0.
    return MNSb * KW
    
    
def Mej_Foucart20_lambda(MBH,MNS,chi,i_tilt,lamb):
    """
    Mass M_ej [Msun] of ejecta. 
    Function of lambda parameter of NS (dimensionless quadrupolar tidal deformability).
    Input:
        MBH    : BH mass [Msun]
        MNS    : NS mass [Msun]
        chi    : BH dimensionless spin parameter [-]
        i_tilt : tilt angle between BH spin and orbit plane [rad] 
        lamb = lambda parameter of NS [-]	
    """
    #Foucart constants
    a1 = 0.007116
    a2 = 0.001436
    a4 = -0.02762
    n1 = 0.8636
    n2 = 1.6840
    Q = MBH/MNS
    
    C = compactness(lamb)
    MNSb = m_to_mb(MNS,C)   
        
    risco = BH_prop.R_isco(chi*np.cos(i_tilt),MBH,'no')
    KW = a1*(Q**n1)*(1.-2.*C)/C - a2*(Q**n2)*risco + a4
    if np.isscalar(KW):
        if KW>1.:
            KW = 1.
        elif KW<0.:
            KW = 0.
    else:
        KW[KW>1.]=1.
        KW[KW<0.]=0.
    return MNSb * KW


def vej_Kawaguchi16(MBH,MNS):
    """
    Ejecta velocity v_ej [c].
    Input: 
      MBH    : BH mass [Msun]
      MNS    : NS mass [Msun]
    """
    v = 0.01533*(MBH/MNS)+0.1907
    return 0.01533*(MBH/MNS)+0.1907    


def m_to_mb(mns,comp):
    return mns * ( 1. + (0.6 * comp / (1. - 0.5 * comp)) )


def compactness(lamb):
	return 0.36 - 0.0355*np.log(lamb) + 0.000705*np.log(lamb)**2
	    


def mdyn_fit_Kruger(mbh,mns,chi,i_tilt,lamb):
    a1 = 0.007116
    a2 = 0.001436
    a4 = -0.02762
    n1 = 0.8636
    n2 = 1.6840    
    comp = compactness(lamb)
    mnsb = m_to_mb(mns,comp) 
    risco = BH_prop.R_isco(chi*np.cos(i_tilt),mbh,'no')
    Q = mbh/mns
    val = a1 * Q**n1 * (1. - 2.*comp)/comp - a2 * Q**n2 * risco + a4
    return val * mnsb

def mdyn_fit_Kruger2(mbh,mns,chi,i_tilt,lamb):
    a1 = 0.007116
    a2 = 0.001436
    a4 = -0.02762
    n1 = 0.8636
    n2 = 1.6840    
    comp = compactness(lamb)
    mnsb = m_to_mb(mns,comp) 
    risco = BH_prop.R_isco(chi*np.cos(i_tilt),mbh,'no')
    Q = mbh/mns
    val = a1 * Q**n1 * (1. - 2.*comp)/comp - a2 * Q**n2 * risco + a4
    return val * mnsb