import numpy as np

# UNIT_LUM = 1e44
UNIT_LUM = 1.0
SIGMA = 6.304e-18

def QVacca(mass):
    """ 
    Input: mass in solar mass
    Return: VaccaQHI in 1e44 s^-1. 
    
    Mass range: 1 - 10000. 
    
    """

    # s**(-1) then normalised in code units in read_stellar
    stf_k=9.634642584812752e48
    # Msun then normalised in code units in read_stellar
    stf_m0=2.728098824280431e1
    stf_a=6.840015602892084e0
    stf_b=4.353614230584390e0
    stf_c=1.142166657042991e0
    return 1. / UNIT_LUM * stf_k * (mass /stf_m0)**stf_a / (1 + (
            mass/stf_m0)**stf_b)**stf_c


def poly(logm, a):
    ret = 0.0
    for i, ai in enumerate(a):
        ret += ai * logm**i
    return ret


def QHeI(mass):
    """ Return Schaerer QHeI """

    a = [16.05, 48.87, -24.70, 4.29]
    logm = np.log10(mass)
    if mass < 6:
        return 0.
    elif mass < 150:
        return 10 ** poly(logm, a)
    else:
        if mass > 1e4:
            hardness = 0.0
        else:
            hardness1 = poly(np.log10(150), a) - np.log10(QVacca(150))
            hardness2 = 0.0
            x1 = np.log10(150)
            x2 = 4
            hardness = hardness1 + (hardness2 - hardness1) / \
                       (x2 - x1) * (logm - x1)
        return QVacca(mass) * 10**hardness

        
def QHeII(mass):
    """ Return Schaerer QHeII """

    logm = np.log10(mass)
    a = [34.65, 8.99, -1.4]
    x0 = np.log10(150)
    if mass < 6:
        return 0.
    elif mass < 150:
        return 10 ** poly(logm, a)
    elif mass < 7e4:
        return 10 ** (poly(x0, a) + (logm - x0) * (a[1] + 2*a[2]*x0))
    else:
        return QVacca(mass)


def sigmaHI(nu):
    """ [nu] = eV
    Return sigma (cm^2)
    """

    sigma = 0.0
    if nu > 13.6 - 1e-10:
        sigma = SIGMA * (nu / 13.6)**-3
    return sigma

def sigmaHeI(nu):
    """ [nu] = eV """

    sigma = 0.0
    epsilon = 1e-10
    if nu > 65.4 - epsilon:
        sigma = sigmaHI(nu) * (37.0 - 19.1 * (nu / 65.4)**(-0.76))
    if nu > 24.6 - epsilon:
        sigma = sigmaHI(nu) * (6.53 * (nu / 24.6) - 0.22)
    return sigma

def sigmaHeII(nu):
    """ [nu] = eV """

    sigma = 0.0
    if nu > 54.4 - 1e-10:
        sigma = SIGMA / 4 * (nu / 13.6 / 4)**-3
    return sigma


def mass_to_lifetime_scaler(mass):
    """ Schaller et al. (1992). (See also MBW Eq. 10.72)

    Input: mass in solar mass
    Return: lifetime in Myr

    """

    lf_max = 1.4e4
    nume = 2.5e3 + 6.7e2 * mass**2.5 + mass**4.5
    deno = 3.3e-2 * mass**1.5 + 3.5e-1 * mass**4.5
    lftm = nume / deno
    # return min(lftm, lf_max)
    return lftm

mass_to_lifetime = np.vectorize(mass_to_lifetime_scaler)
