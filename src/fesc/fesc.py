import os
import sys
from math import sin, cos, log
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as pl
from pymses import RamsesOutput
from pymses.analysis import sample_points
from pymses.utils import constants as C
import argparse

from .utils import QVacca, mass_to_lifetime 

# DEBUG, 0: no debug, 1: debug with uniform density, 2: debug with RAMSES data
DEBUG = 0


Sigma_HI = 6.304e-18    # cm^2
MH = 1.6733e-24         # g
Kappa_HI = Sigma_HI / MH
Kappa_HeI = Kappa_HI * (24.6 / 13.6)**-3
Kappa_HeII = Kappa_HI * (54.4 / 13.6)**-3
OPACITIES = {"HI": Kappa_HI, "HeI": Kappa_HeI, "HeII": Kappa_HeII}


def col_den_all_stars_and_directions(ro, star_pos: np.ndarray, n_angular_refine, nsample, H_fraction, He_fraction, seed=None):
    """
    Args:
    -----
    ro: pymses.RamsesOutput
        The RamsesOutput object
    stars: np.ndarray
        The stars positions normalized to [0, 1] across the simulation box. The shape is (nstars, 3)
    n_angular_refine: int
        The total number of pixels will be 12 * 4^n_angular_refine.
    nsample: int
        The number of Monte Carlo sampling points
        
    Returns:
    --------
    colDenHI: np.ndarray
        The neutral hydrogen column density. The shape is (nstars, Npixel). 
        The unit is g/cm^2.
    colDenHeI: np.ndarray
        The neutral helium column density. The shape is (nstars, Npixel)
    colDenHeII: np.ndarray
        The helium II column density. The shape is (nstars, Npixel)
    
    """

    assert star_pos.ndim == 2, "stars must be a numpy array with 2 dimensions"
    assert star_pos.shape[1] == 3, "stars must be a numpy array with 2 dimensions, and the second dimension must be 3"

    if seed is not None:
        np.random.seed(seed)

    # unit
    t = ro.info['time'] * ro.info['unit_time'].express(C.Myr)
    boxlen = np.double(ro.info['boxlen'])
    unit_l = ro.info['unit_length']
    unit_d = ro.info['unit_density']
    unit_col_den = (unit_l * unit_d).express(C.cm**(-2) * C.g)
    nstars = star_pos.shape[0]

    # print("\nCalculating the following snapshot:")
    # print(f"jobdir = {ro.output_repos}")
    # print(f"iout = {ro.iout}")
    # print(f"unit_l = {unit_l}, unit_d = {unit_d}, unit_col_den = {unit_col_den} cm-2 g")
    # print(f"time = {ro.info['time']} = {t} Myr")
    # print(f"Number of stars = {nstars}")

    # star_surf = 5.1 * 1. / 2**level_refine
    star_surf = 0.0
    # radius_end = .5  # boxlen = 1, radius = 0.5
    # radius_end = 1.0            # v6
    # radius_end = 0.5            # v7
    radius_end = 1.0            # v10

    # For debugging purpose only
    if DEBUG > 0:
        radius_end = 0.1

    source = ro.amr_source(["rho", "xHII", "xHeII", "xHeIII"])

    # healpy stuff
    nside = 2**n_angular_refine            # nside = 2**integer
    Npixel = hp.nside2npix(nside)   # 12 * nside**2
    angles = pl.pix2ang(nside, np.arange(Npixel))
    angles = np.array(angles).transpose()  # [*, 2]
    theta = angles[:, 0]
    phi = angles[:, 1]

    # count total sample points
    tot_sample = nstars * Npixel * nsample

    # output container
    meanDenHI = np.zeros((nstars, Npixel))  # Neutral H column density
    meanDenHeI = np.zeros((nstars, Npixel))  # Neutral He column density
    meanDenHeII = np.zeros((nstars, Npixel))  # Column density of HeII

    maximum_points = int(1e8) # total memory: 800 MB
    num_of_stars_per_loop = min(int(maximum_points / (Npixel * nsample)), nstars)
    assert num_of_stars_per_loop > 0, (f"num_of_star_per_loop must be greater than 0. nstars "
                                       "= {nstars}, Npixel = {Npixel}, nsample = {nsample}")

    count_start = 0
    count_end = num_of_stars_per_loop
    while count_start < nstars:
        num_of_stars_in_group = count_end - count_start

        points = np.zeros([num_of_stars_in_group, Npixel, nsample, 3])

        # loop over stars
        for i_star in range(num_of_stars_in_group):
            i_star_global = count_start + i_star
            rad_center = star_pos[i_star_global, :]
            # loop over directions
            for i_pixel in range(Npixel):
                itheta = theta[i_pixel]
                iphi = phi[i_pixel]
                radius = np.random.random(nsample) * (radius_end - star_surf) + star_surf
                points[i_star, i_pixel, :, 0] = radius * sin(itheta) * cos(iphi) + rad_center[0]
                points[i_star, i_pixel, :, 1] = radius * sin(itheta) * sin(iphi) + rad_center[1]
                points[i_star, i_pixel, :, 2] = radius * cos(itheta) + rad_center[2]
        
        # reshape points
        tot_points_in_group = num_of_stars_in_group * Npixel * nsample
        dots = points.reshape((tot_points_in_group, 3))

        # read the hydro data
        if DEBUG != 1:
            sys.stdout = open(os.devnull, "w")
            sp = sample_points(source, dots)
            rho = sp.fields['rho']
            xHII = sp.fields['xHII'] * H_fraction
            xHeII = sp.fields['xHeII'] * He_fraction
            xHeIII = sp.fields['xHeIII'] * He_fraction
            sys.stdout = sys.__stdout__
        else:
            # test 1
            # rho = np.ones(tot_points_in_group) * 1e-3
            # test 2
            # rho = np.ones(tot_points_in_group) * 1e-3
            # rho[20*nsample:21*nsample] = 1e9
            # test 3
            rho = np.ones(tot_points_in_group) * 1e-3
            if count_start == 0:
                rho[20*nsample:21*nsample] = 1e9
            xHII = np.zeros(tot_points_in_group)
            xHeII = np.zeros(tot_points_in_group)
            xHeIII = np.zeros(tot_points_in_group)

        assert xHII.max() <= 1.0, f"xHII.max() = {xHII.max()}"
        assert xHII.min() >= 0.0, f"xHII.min() = {xHII.min()}"
        xHI = H_fraction - xHII
        xHeI = He_fraction - xHeII - xHeIII

        # Set the density of the dots outside the box to 0
        for i in range(3):
            rho[dots[:, i] < 0] = 0
            rho[dots[:, i] > 1] = 0

        # calculate column density
        for i_star in range(num_of_stars_in_group):
            i_star_global = count_start + i_star
            for i_pixel in range(Npixel):
                idx_1 = (i_star * Npixel + i_pixel) * nsample
                idx_2 = (i_star * Npixel + i_pixel + 1) * nsample
                meanDenHI[i_star_global, i_pixel] = np.mean(rho[idx_1:idx_2] * xHI[idx_1:idx_2])
                meanDenHeI[i_star_global, i_pixel] = np.mean(rho[idx_1:idx_2] * xHeI[idx_1:idx_2])
                meanDenHeII[i_star_global, i_pixel] = np.mean(rho[idx_1:idx_2] * xHeII[idx_1:idx_2])

        count_start = count_end
        count_end = min(count_end + num_of_stars_per_loop, nstars)

    # length = boxlen * (radius_end - star_surf)
    length = (radius_end - star_surf)
    coeff = length * unit_col_den
    return meanDenHI * coeff, meanDenHeI * coeff, meanDenHeII * coeff
        

def compute_angular_fesc(col_den: np.ndarray, kappa: float):
    """
    Parameters:
    -----------
    col_den: np.ndarray
        The column density. The shape is (Npixel,)
    kappa: float
        The opacity of the medium. The unit is cm^2/g
    
    """

    tau = col_den * kappa
    fesc = 1.0 - np.exp(-tau)
    return fesc

    
def compute_weighted_fesc(tau, weights):
    """
    
    Parameters
    ----------
    tau: np.ndarray
        The optical depth. The shape is (Nstars, Npixel)
    weights: np.ndarray
        The photon emission weights of the stars. The shape is (Nstars,)
    
    Returns
    -------
    fesc_weighted: float
        The weighted escape fraction.
    
    Examples
    --------
    
    """
    
    fesc = np.exp(-tau)
    fesc_mean = np.mean(fesc, axis=1)
    fesc_weighted = np.sum(fesc_mean * weights) / np.sum(weights)
    return fesc_weighted


def plot_sky(fesc, vmin=-2, vmax=0, is_log=True, fn="./sky"):
    
    if is_log:
        z = np.log10(fesc)
    else:
        z = fesc

    sky_im_arr = hp.mollview(z, min=vmin, max=vmax, return_projected_map=1)
    plt.savefig(f"{fn}-healpy.png", dpi=300)

    f, ax = plt.subplots()
    cmap = plt.cm.viridis
    cmap.set_under('k')
    # cmap.set_bad('k')
    im = ax.imshow(sky_im_arr, cmap=cmap, vmin=vmin, vmax=vmax)

    # adjust ax and add a colorbar on the bottom
    f.subplots_adjust(bottom=0.1, top=0.98, left=0.02, right=0.98)
    # hide the axes
    ax.axis('off')
    cax = f.add_axes([0.2, 0.15, 0.6, 0.02])
    cb = f.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize='small')
    cb.ax.minorticks_on()
    cb.set_label(r"$\log_{10} f_{\rm esc}(\vec{\theta})$", fontdict={'size':'medium'},)
    f.savefig(f"{fn}-matplotlib.png", dpi=300)
    return


def arg_parser():

    parser = argparse.ArgumentParser(
        description="Calculate the column density in all directions from a single star")
    parser.add_argument("jobdir", type=str, help="The simulation directory")
    # parser.add_argument("outputID", type=int, help="output_#####")
    # parser.add_argument("starID", type=int, help="The star ID")
    parser.add_argument("nsample", type=int, help="Number of Monte Carlo sampling points")
    parser.add_argument("refine", type=int, help="The number of spatial directions will be 12 * 4^refine")
    # parser.add_argument("all_H", type=bool, help="False: calculate the neutral hydrogen column density")
    return parser.parse_args()    

