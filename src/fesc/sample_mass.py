#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A quick demo of sampling from a power-law IMF, given the total mass in stars. 
Algorithm design:
1. Given a IMF, calculate the fraction of mass in stars between m_min and m_max, mass_frac_in_massive_stars.
2. Calculate the average mass of stars in this mass range, m_ave.
Loop over all positions in the simulation:
    3. Calculate the expected number of stars in this mass range at a given position, n_ave = m_particle * mass_frac_in_massive_stars / m_ave.
    4. Sample the number of stars from a Poisson distribution with the mean calculated in step 3, n_star.
    5. Sample n_star stars, each star is drawn from the IMF between m_min and m_max.

@author: Chong-Chong He
"""

import numpy as np

def sample_powerlaw_imf(m_min, m_max, alpha, N):
    """
    Sample N masses from a power law distribution p(m) ∝ m^α between m_min and m_max.
    
    Parameters
    ----------
    m_min : float
        Minimum mass
    m_max : float
        Maximum mass
    alpha : float
        Power law index (must < -1)
    N : int
        Number of samples to draw
        
    Returns
    -------
    masses : array
        Array of N sampled masses
    """
    if alpha >= -1:
        raise ValueError("alpha must be < -1 for a power law IMF")
        
    # For p(m) ∝ m^α, the CDF is:
    # P(m) = (m^(α+1) - m_min^(α+1)) / (m_max^(α+1) - m_min^(α+1))
    # The inverse CDF is:
    # m = [P * (m_max^(α+1) - m_min^(α+1)) + m_min^(α+1)]^(1/(α+1))
    
    # Generate uniform random numbers
    P = np.random.uniform(0, 1, N)
    
    # Calculate masses using inverse CDF
    alpha_plus_1 = alpha + 1
    m_min_alpha_plus_1 = m_min ** alpha_plus_1
    m_max_alpha_plus_1 = m_max ** alpha_plus_1
    masses = (P * (m_max_alpha_plus_1 - m_min_alpha_plus_1) + m_min_alpha_plus_1) ** (1/alpha_plus_1)
    
    return masses


def average_mass(m_min, m_max, alpha):
    """
    Calculate the average mass of a power law IMF.
    = (a + 1) / (a + 2) * (b^(a+2) - a^(a+2)) / (b^(a+1) - a^(a+1))
    """
    return (alpha + 1) / (alpha + 2) * (m_max ** (alpha + 2) - m_min ** (alpha + 2)) / (m_max ** (alpha + 1) - m_min ** (alpha + 1))


def random_number_from_poisson(n_average, N):
    """
    Sample N random numbers from a Poisson distribution with mean n_average.
    """
    return np.random.poisson(n_average, N)


# test
def test_distribution():

    #------ test power law distribution ------
    m_min = 1
    m_max = 200
    alpha = -2.3
    N = 10000
    masses = sample_powerlaw_imf(m_min, m_max, alpha, N)
    assert np.all(masses >= m_min) and np.all(masses <= m_max)
    m_mean = np.mean(masses)
    m_mean_expected = average_mass(m_min, m_max, alpha)
    assert np.isclose(m_mean, m_mean_expected, atol=0.1), f"m_mean = {m_mean}, m_mean_expected = {m_mean_expected}"

    # # plot
    # import matplotlib.pyplot as plt
    # # from imf import plot_imf

    # f, ax = plt.subplots()
    # plot_imf(ax, masses, bins='auto', xlim=[-1, 3], ylim=[1, 1e5])
    # plt.savefig('powerlaw_imf.png')

    #------ test Poisson distribution ------
    n_average = 1
    n_samples = 1000
    dist = random_number_from_poisson(n_average, n_samples)
    print(dist[:100])
    print(np.mean(dist))
    assert np.isclose(np.mean(dist), n_average, atol=0.1), f"np.mean(dist) = {np.mean(dist)}, n_average = {n_average}"


def sample_massive_stars(N, m_particle=10.0):

    m_min = 8
    m_max = 120
    alpha = -2.35
    mass_frac_in_massive_stars = 3.638945e-01 # the fraction of mass above 8 M_sun for a Chabrier IMF (0.08 M_sun < m < 120 M_sun)
    m_mean_massive_stars = average_mass(m_min, m_max, alpha)
    n_average = m_particle * mass_frac_in_massive_stars / m_mean_massive_stars

    dist = random_number_from_poisson(n_average, N)

    m_collection = []
    for n in dist:
        if n == 0:
            m_collection.append([])
        else:
            m_sample = sample_powerlaw_imf(m_min, m_max, alpha, n)
            m_collection.append(m_sample)
    return m_collection


def do_sample(m_particle=10.0):

    # solar mass. this the stellar mass at each position in the simulation
    m_min = 8
    m_max = 120
    alpha = -2.35
    mass_frac_in_massive_stars = 3.638945e-01 # the fraction of mass above 8 M_sun for a Chabrier IMF (0.08 M_sun < m < 120 M_sun)
    m_mean_massive_stars = average_mass(m_min, m_max, alpha)
    n_average = m_particle * mass_frac_in_massive_stars / m_mean_massive_stars

    N = 400
    dist = random_number_from_poisson(n_average, N)
    N_per_group = 10
    N_group = N // N_per_group

    print(f"""
    Doing a sampling with the following parameters:
    A total of {N} positions in the simulation, each position has a stellar mass of {m_particle}.
    mass fraction in massive stars (from {m_min} to {m_max}): {mass_frac_in_massive_stars} (this is just a random number I made up. should depend on the exact IMF).
    power-law IMF: m_min = {m_min}, m_max = {m_max}, alpha = {alpha}
    mean mass of massive stars in a sampled population: {m_mean_massive_stars}
    average number of massive stars at a position: n_average = {n_average}
    """)

    print(f"Number of stars in each position, a total of {N} positions. This is a Poisson distribution with mean n_average.")
    for i in range(N_group):
        print(dist[i*N_per_group:(i+1)*N_per_group])
    print()

    def tmp(x):
        print(x, end=' ')
        return

    print(f"The mass of massive stars sampled:")
    m_collection = []
    for i in range(N_group):
        for n in dist[i*N_per_group:(i+1)*N_per_group]:
            if n == 0:
                tmp(n)
            else:
                m_sample = sample_powerlaw_imf(m_min, m_max, alpha, n)
                m_collection.extend(m_sample)
                tmp(m_sample)
        print()
    
    print(f"\ntotal mass in sampled massive stars = {np.sum(m_collection)} (should be near {m_particle * mass_frac_in_massive_stars * N})")


if __name__ == "__main__":

    # test_distribution()
    # do_sample()

    print(sample_massive_stars(10))
