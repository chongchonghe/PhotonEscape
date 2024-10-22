import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymses import RamsesOutput
from pymses.analysis import sample_points
from time import time
from pymses.utils import constants as C
import argparse
import yt
from yt.funcs import mylog
from unyt import cm, unyt_quantity
mylog.setLevel(40)

from fesc import fesc

DEBUG = 0
fesc.DEBUG = DEBUG
INCLUDE_He = 0

ONLY_ALIVE = 1
MAX_AGE = 30.0 # Myr

def get_npz_filename(out_num, ref_level):
    return f"colden_{out_num:05d}_level{ref_level}.npz"

# from Fred's code: tools/cosmos.py
def code_age_to_myr(all_star_ages, hubble_const, unique_age=True, true_age=False):
    r"""
    Returns an array with unique birth epochs in Myr given
    raw_birth_epochs = ad['star', 'particle_birth_epoch']
    AND
    hubble = ds.hubble_constant
    Youngest is 0 Myr, all others are relative to the youngest.

    Relative ages option is currently yielding inconsistent results
    """
    cgs_yr = 3.1556926e7  # 1yr (in s)
    cgs_pc = 3.08567758e18  # pc (in cm)
    h_0 = hubble_const * 100  # hubble parameter (km/s/Mpc)
    h_0_invsec = h_0 * 1e5 / (1e6 * cgs_pc)  # hubble constant h [km/s Mpc-1]->[1/sec]
    h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]

    if unique_age is True:
        # process to unique birth epochs only as well as sort them
        be_star_processed = np.array(sorted(list(set(all_star_ages.to_ndarray()))))
        star_age_myr = (be_star_processed * h_0inv_yr) / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()
    else:
        all_stars = all_star_ages
        star_age_myr = all_stars * h_0inv_yr / 1e6  # t=0 is the present
        relative_ages = star_age_myr - star_age_myr.min()
    if true_age is True:
        return star_age_myr  # + 13.787 * 1e3
    else:
        return relative_ages  # t = 0 is the age of

def get_star_ages(ram_ds, ram_ad, logsfc):
    """
    star's ages in [Myr]
    """
    # first_form = np.loadtxt(logsfc, usecols=2).max()
    # hack by CCH: set formation redshift to 30
    first_form = 30
    current_hubble = ram_ds.hubble_constant
    current_time = float(ram_ds.current_time.in_units("Myr"))

    birth_start = np.round_(
        float(ram_ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
    )
    converted_unfiltered = code_age_to_myr(
        ram_ad["star", "particle_birth_epoch"],
        current_hubble,
        unique_age=False,
    )
    birthtime = np.round(converted_unfiltered + birth_start, 3)  #!
    current_ages = np.array(np.round(current_time, 3) - np.round(birthtime, 3))

    # hack by CCH: set the youngest star to 0 Myr
    current_ages -= current_ages.min()

    return current_ages



def arg_parser():

    parser = argparse.ArgumentParser(
        description="Calculate the column density in all directions from a single star")
    parser.add_argument("task", type=str, help="Task to do: 'process', 'fesc', or 'process_and_fesc'")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="The simulation data directories")
    parser.add_argument("--outdir", type=str, default="outs", help="the directory for output figures.")
    parser.add_argument("--nsample", type=int, nargs="?", default=100, help="Number of Monte Carlo sampling points for each light beam. Default: 100")
    parser.add_argument("--refine", type=int, nargs="?", default=0, help="The number of spatial directions will be 12 * 4^refine. Default: 0")
    parser.add_argument("--dist", type=str, nargs="?", default="1_kpc", help="The distance to do ray tracing. Dimensionless numbers will be the fraction to the box size. Default: 1. Examples: '1', '1.5_kpc'")
    parser.add_argument("--subsample", type=float, default=0.01, help="Sub-sampling fraction for the star particles. Default: 1.0")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="The maximum number of samples to do per process. The default is 1e8, which results in 3.2 GB memory usage per process. Increasing this number will increase the speed but also the memory usage linearly.")
    return parser.parse_args()


def get_all_outputs(outputs):
    all_outputs = []
    for input_dir in outputs:
        if os.path.basename(input_dir).startswith("output_"):
            try:
                out_num = int(input_dir.split("_")[-1])
                input_base_dir = os.path.dirname(input_dir)
                all_outputs.append((input_base_dir, out_num))
            except:
                pass
    return all_outputs


def process_outputs(args):

    cell_fields = [
        "Density",
        "x-velocity",
        "y-velocity",
        "z-velocity",
        "Pressure",
        "Metallicity",
        # "dark_matter_density",
        "xHI",
        "xHII",
        "xHeII",
        "xHeIII",
    ]
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]

    def tau_to_fesc(tau):
        return np.exp(-tau)

    all_outputs = get_all_outputs(args.input)
    for the_output in all_outputs:
        input_base_dir, out_num = the_output

        infopath = f"{input_base_dir}/output_{out_num:05d}/info_{out_num:05d}.txt"
        ds = yt.load(infopath, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        #----- Read star positions and masses  -----
        star_mass = ad["star", "particle_mass"].in_units("Msun")
        star_pos = ad["star", "particle_position"].in_units("code_length")
        star_pos = star_pos.value
        star_age = get_star_ages(ds, ad, None)
        n_star = len(star_mass)
        print(f"Found {n_star} stars")
        if n_star == 0:
            continue

        assert args.subsample <= 1.0, "Error: subsample must be less than or equal to 1.0"
        if args.subsample < 1.0:
            n_star_pick = int(n_star * args.subsample)
            # pick n_star_pick integers from 0 to n_star-1 
            idx = np.random.choice(n_star, n_star_pick, replace=False)
            star_mass = star_mass[idx]
            star_pos = star_pos[idx, :]
            star_age = star_age[idx]

        t = ds.current_time.in_units("Myr")
        print(f"\nProcessing snapshot {out_num:05d}, current time: {t:.2f} Myri, number of stars processed: {len(star_mass)}")

        # parse the distance: if it is a number, it is the fraction of the box size, otherwise it is number_unit
        try:
            dist = float(args.dist)
        except:
            assert args.dist.count("_") == 1, "Error: distance must be a number or a number_unit"
            quant = args.dist.replace("_", " ")
            dist = unyt_quantity.from_string(quant) / ds.domain_width[0]
            dist = dist.value
        
        #----- compute column density  -----
        ro = RamsesOutput(input_base_dir, out_num)
        sampleNum = args.nsample
        refine = args.refine
        n_directions = 12 * 4**refine
        print("Processing output", out_num, "with", sampleNum, "samples and", n_directions, "angular directions")
        t0 = time()
        col_H2, col_HI, col_HeI, col_HeII = fesc.col_den_all_stars_and_directions(
            ro, star_pos, n_angular_refine=refine, nsample=sampleNum, H_fraction=0.76, He_fraction=0.24, 
            ray_start=0.0, ray_end=dist, seed=333)
        dt = time() - t0
        print(f"Done processing. Time taken = {dt:.2f} s")

        #----- save the results  -----
        outdir = f"{input_base_dir}/processed/colden"
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, get_npz_filename(out_num, refine))
        # save column density for all stars and all directions, star masses (Msun), and star positions (code_length)
        np.savez(outpath, col_H2=col_H2, col_HI=col_HI, col_HeI=col_HeI, col_HeII=col_HeII, star_mass=star_mass, star_pos=star_pos, star_age=star_age)

        # #----- compute the escape fraction for all directions from all stars  -----
        # # The dimensions of tau_HI are (n_star, n_directions)
        # kappa_HI = fesc.OPACITIES["HI"]
        # tau_HI = col_1 * kappa_HI
        # fesc_HI = np.exp(-tau_HI)
        # kappa_HeI = fesc.OPACITIES["HeI"]
        # tau_HeI = col_2 * kappa_HeI
        # fesc_HeI = np.exp(-tau_HeI)
        # kappa_HeII = fesc.OPACITIES["HeII"]
        # tau_HeII = col_3 * kappa_HeII
        # fesc_HeII = np.exp(-tau_HeII)
        
        # #----- save the results  -----
        # outdir = f"{jobdir}/processed/fesc"
        # os.makedirs(outdir, exist_ok=True)
        # outpath = f"{outdir}/fesc_{out:05d}.npz"
        # # save: the escape fraction for all stars and all directions; star masses (Msun); star positions (code_length)
        # np.savez(outpath, fesc_HI=fesc_HI, fesc_HeI=fesc_HeI, fesc_HeII=fesc_HeII, star_mass=star_mass, star_pos=star_pos, star_age=star_age)
    
    return


def compute_fesc(args):

    out_dir = "."
    if args.outdir is not None:
        out_dir = args.outdir
        os.makedirs(out_dir, exist_ok=True)

    if args.refine is None:
        sys.exit("Error: refine must be specified")
    refine = args.refine

    all_outputs = get_all_outputs(args.input)
    for the_output in all_outputs:
        input_base_dir, out_num = the_output
        outpath = os.path.join(input_base_dir, "processed/colden", get_npz_filename(out_num, refine))
        if not os.path.exists(outpath):
            print(f"Error: {outpath} does not exist. Do data process first. Skipping...")
            continue
        data = np.load(outpath)
        col_H2 = data["col_H2"] # shape: (n_star, n_directions)
        col_HI = data["col_HI"] # shape: (n_star, n_directions)
        col_HeI = data["col_HeI"]
        col_HeII = data["col_HeII"]
        star_mass = data["star_mass"]
        star_pos = data["star_pos"]
        star_age = data["star_age"]
        n_star = col_HI.shape[0]

        is_alive = np.array(star_age < MAX_AGE, dtype=int)      # shape: (n_star,)

        #----- compute the escape fraction for all directions from all stars  -----
        # The dimensions of tau_HI are (n_star, n_directions)
        tau_HI = fesc.compute_HI_ion_optial_depth(col_H2, col_HI, col_HeI, col_HeII)
        fesc_HI = np.exp(-tau_HI)
        if INCLUDE_He:
            tau_HeI = col_HeI * fesc.OPACITIES["HeI"]
            fesc_HeI = np.exp(-tau_HeI)
            tau_HeII = col_HeII * fesc.OPACITIES["HeII"]
            fesc_HeII = np.exp(-tau_HeII)
        
        if not ONLY_ALIVE: # all stars 
            fesc_HI_star_mean = np.mean(fesc_HI, axis=0)        # shape: (n_directions,)
            if INCLUDE_He:
                fesc_HeI_star_mean = np.mean(fesc_HeI, axis=0)      # shape: (n_directions,)
                fesc_HeII_star_mean = np.mean(fesc_HeII, axis=0)    # shape: (n_directions,)
        else: # only alive stars
            is_alive_expanded = is_alive[:, None]
            fesc_HI_star_mean = np.mean(fesc_HI * is_alive_expanded, axis=0)
            if INCLUDE_He:
                fesc_HeI_star_mean = np.mean(fesc_HeI * is_alive_expanded, axis=0)
                fesc_HeII_star_mean = np.mean(fesc_HeII * is_alive_expanded, axis=0)

        fesc_HI_star_and_sky_mean = np.mean(fesc_HI_star_mean)
        fesc_HI_star_and_sky_std = np.std(fesc_HI_star_mean)
        fesc_HeI_star_and_sky_mean = None
        fesc_HeII_star_and_sky_mean = None
        fesc_HeI_star_and_sky_std = None
        fesc_HeII_star_and_sky_std = None
        if INCLUDE_He:
            fesc_HeI_star_and_sky_mean = np.mean(fesc_HeI_star_mean)
            fesc_HeII_star_and_sky_mean = np.mean(fesc_HeII_star_mean)
            fesc_HeI_star_and_sky_std = np.std(fesc_HeI_star_mean)
            fesc_HeII_star_and_sky_std = np.std(fesc_HeII_star_mean)

        print(f"Output {out_num:05d} with {n_star} stars:")
        print("Sky-mean escape fraction for HI, HII, HeII:", fesc_HI_star_and_sky_mean, fesc_HeI_star_and_sky_mean, fesc_HeII_star_and_sky_mean)
        print("Sky-standard deviation for HI, HII, HeII:", fesc_HI_star_and_sky_std, fesc_HeI_star_and_sky_std, fesc_HeII_star_and_sky_std)

        #----- plot the sky map: luminosity weighted escape fraction -----
        # luminosity = fesc.QVacca(star_mass)
        # weights = luminosity * 1e-44
    
        # plot HI
        weights = np.ones(len(star_mass))
        fesc1_sky_weighted = np.dot(weights, fesc_HI) / np.sum(weights)
        fesc.plot_sky(fesc1_sky_weighted, vmin=-6, vmax=0, is_log=True, fn=f"{out_dir}/sky-output{out_num:05d}-HI", axis_on=0)

        if INCLUDE_He:
            # plot HeI
            fesc2_sky_weighted = np.dot(weights, fesc_HeI) / np.sum(weights)
            fesc.plot_sky(fesc2_sky_weighted, vmin=-6, vmax=0, is_log=True, fn=f"{out_dir}/sky-output{out_num:05d}-HeI", axis_on=0)
            
            # plot HeII
            fesc3_sky_weighted = np.dot(weights, fesc_HeII) / np.sum(weights)
            fesc.plot_sky(fesc3_sky_weighted, vmin=-6, vmax=0, is_log=True, fn=f"{out_dir}/sky-output{out_num:05d}-HeII", axis_on=0)

        print(f"Sky map saved to {out_dir}/sky-output{out_num:05d}-xx.png")


if __name__ == "__main__":

    Args = arg_parser()
    if Args.task == "process":
        process_outputs(Args)
    elif Args.task == "fesc":
        compute_fesc(Args)
    elif Args.task == "process_and_fesc":
        process_outputs(Args)
        compute_fesc(Args)
    else:
        sys.exit("Error: task must be either 'process' or 'fesc' or 'process_and_fesc'")
    print("\nDone.")
