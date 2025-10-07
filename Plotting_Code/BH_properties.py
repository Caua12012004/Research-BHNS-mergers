import numpy as np
import units as u

# Functions of Spin to calculate R_isco
def Z1(x):
    """
    Input: x = spin parameter [-]
    """
    return(1.+np.power(1.-x**2,1./3.)*(np.power(1.+x,1./3.)+np.power(1.-x,1./3.)))

def Z2(x):
    """
    Input: x = spin parameter [-]
    """	
    return(np.sqrt(3.*(x**2)+Z1(x)**2))	


def R_isco(x,m,units):
    """
    Innermost Stable Circular Orbit
    Input: 
      x    : dimensionless spin paramater [-]
      m    : BH mass [Msun]
      units: desired output units ['km','cm' or 'no'(output in gravitational radii)]
    Output: 
      R_isco [km/cm/Gravitational Radii (GM/c^2)]
    """
    if (units=='km'):
        R_g = m*u.G*u.Msun/u.c**2/1.e5
        return((3.+Z2(x)-np.sign(x)*np.sqrt((3.-Z1(x))*(3.+Z1(x)+2.*Z2(x))))*R_g)
    if (units=='cm'):
        R_g = m*u.G*u.Msun/u.c**2
        return((3.+Z2(x)-np.sign(x)*np.sqrt((3.-Z1(x))*(3.+Z1(x)+2.*Z2(x))))*R_g)
    if (units=='no'):
        return((3.+Z2(x)-np.sign(x)*np.sqrt((3.-Z1(x))*(3.+Z1(x)+2.*Z2(x)))))        		    

def bh_spin(chi,M, **kwargs):
    ''' 
    function to compute the BH spin in natural units
    Input:
      chi   : dimensionless spin paramater [-]
      M     : BH mass [Msun]
    Output:
	  BH_spin : BH spin in natural units [c/G]
    '''
    return chi*M*M