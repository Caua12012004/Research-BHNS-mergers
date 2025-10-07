
import numpy as np
import EOS_dictionary
#import EOS_dictionary 

import units
from scipy.interpolate import interp1d 
from sys import exit 

class NS_EOS_properties(object):

    def __init__(self,EOS_name):

        self.EOS_filename = None

        # load the EOS dictionary
        EOS_list = EOS_dictionary.EOS_list

        # define the file corresponding to the EOS
        not_found = True
        for EOS_name_dict in EOS_list.keys():
            if(EOS_list[EOS_name_dict]['name']==EOS_name):
                self.EOS_filename = r"C:\Users\cauar\OneDrive\Área de Trabalho\Research_Floor\Plotting_Code\\"+EOS_list[EOS_name_dict]['file']
                #print('I am using: ',self.EOS_filename)
                not_found = False
        if (not_found):
            print('Invalid EOS name!')
            exit()

        # TODO:
        # add maximum masses to the EOS dictionary

        # load the EOS file
        gm,bm,r,kl = np.loadtxt(self.EOS_filename,unpack=True,usecols=(1,2,3,4),skiprows=1)

        # compute the compactness from mass and radii
        C = gm/r

        # interpolate all the relevant quantities as a function of the gravitational mass
        self.f_m2mb = interp1d(gm,bm,fill_value='extrapolate')
        self.f_mb2m = interp1d(bm,gm,fill_value='extrapolate')
        self.f_comp = interp1d(gm,C,fill_value='extrapolate')
        self.f_kappal = interp1d(gm,kl,fill_value='extrapolate')

    def compactness(self,m):
        """
        function to compute the star compactness
        Input:
          m: gravitational mass of the star [M_sun]
        """
        return self.f_comp(m) 

    def kappa_love(self,m):
        """
        function to compute the star Love parameter k (Latin kappa)
        Input:
          m: gravitational mass of the star [M_sun]
        """
        return self.f_kappal(m) 

    def g2b_mass(self,m):
        """
        function to compute the baryonic mass of an isolated NS from the
        Input:
          m: gravitational mass of the star [M_sun]
        """
        return self.f_m2mb(m) 

    def b2g_mass(self,mb):
        """
        function to compute the gravitational mass of an isolated NS
        from the baryonic mass
        Input:
          mb: baryonic mass of the star [M_sun]
        """
        return self.f_mb2m(mb) 

    def fun_geqtp_kappa(self,m1,m2):
        """
        function to compute the gravitoelectric quadrupole tidal 
        parameter of a signle NS in a binary system
        Input:
          m1: gravitational mass of the first star [M_sun]
          m2: gravitational mass of the second star [M_sun]
        """
        x = m1/(m1+m2)         # mass fraction of the first star
        c = self.f_comp(m1)    # compactness of the first star 
        k = self.f_kappal(m1)  # Love's k of the first star (Latin kappa)
        return 2.e0*(1.e0-x)*k/x*(x/c)**5

    def fun_lambda_kappa(self,m):
        """
        function to compute the dimensionless tidal deformability 
        for a single NS ('Capital Lambda')
        Input:
          m: gravitational mass of the first star [M_sun]
        """
        c = self.f_comp(m)    # compactness of the first star 
        k = self.f_kappal(m)  # Love's k of the first star (Latin kappa)
        return (2.e0/3.e0)*k/c**5

    def fun_lambda_tilde(self,ma,mb,la,lb):
        """
        function to compute the dimensionless tidal deformability
        of a binary NS system ('Capital Lambda tilde')
        Input:
          ma: gravitational mass of the first star  [M_sun]
          mb: gravitational mass of the second star [M_sun]
        """
        # compute the dimensionless tidal deformability ('Capital lambda')
        la = self.fun_lambda_kappa(ma)
        lb = self.fun_lambda_kappa(mb)

        mtot5 = (ma+mb)**5
        ma4 = ma**4
        mb4 = mb**4
        tmp1 = (ma+12.e0*mb)
        tmp2 = (mb+12.e0*ma)
        return (16.e0/13.e0)*(tmp1*ma4*la + tmp2*mb4*lb)/mtot5

    # function to compute different kinds of total tidal parameters of a NS binary. Possible choices are:
    #    total      ['tot'] ref:
    #    luminosity ['lum'] ref:
    #    effective  ['eff'] ref:
    def gkappa(self,kind,ma,mb):
        """
        function to compute the total tidal parameter of a NS binary
        ('Greek kappa_2^T')
        Input:
          kind: kind of greek kappa [tot,lum,eff]
          ma  : gravitational mass of the first star  [M_sun]
          mb  : gravitational mass of the second star [M_sun]
        """

        xa = ma/(ma+mb)
        xb = 1.e0-xa

        # compute the gravitoelectric quadrupole tidal polarizabilities
        ga = self.fun_geqtp_kappa(ma,mb) 
        gb = self.fun_geqtp_kappa(mb,ma)

        if (kind == 'tot'):
            return ga+gb
        if (kind == 'lum'):
            return 2.e0*((3.e0-2.e0*xa)*ga/xb + (3.e0-2.e0*xb)*gb/xa )
        if (kind == 'eff'):
            return (2.e0/13.e0)*((1.e0+12.e0*xb/xa)*xa/(2.e0*xb)*ga+(1.e0+12.e0*xa/xb)*xb/(2.e0*xa)*gb)


#######################

if __name__=="__main__":

    NsEosProperties = NS_EOS_properties('SLy')

    m1 = 1.35
    m2 = 1.4
    k = NsEosProperties.fun_geqtp_kappa(m1,m2)

    gk_tot = NsEosProperties.gkappa('tot',m1,m2)
    gk_lum = NsEosProperties.gkappa('lum',m1,m2)
    gk_eff = NsEosProperties.gkappa('eff',m1,m2)

    C1 = NsEosProperties.compactness(m1)
    C2 = NsEosProperties.compactness(m2)
    
    g1 = NsEosProperties.g2b_mass(m1)
    g2 = NsEosProperties.g2b_mass(m2)
    
    print(g1)
    print(g2)
    print(C1)
    print(C2)
    print(gk_tot)
    print(gk_lum)
    print(gk_eff)

    print(k)