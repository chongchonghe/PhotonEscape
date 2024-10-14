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

def arg_parser():

    parser = argparse.ArgumentParser(
        description="Calculate the column density in all directions from a single star")
    parser.add_argument("task", type=str, help="Task to do: 'process' or 'fesc'")
    parser.add_argument("jobdir", type=str, help="The simulation data directory")
    parser.add_argument("--output", type=int, nargs="?", help="the output to process. Default: all outputs in the jobdir")
    parser.add_argument("--nsample", type=int, nargs="?", default=100, help="Number of Monte Carlo sampling points for each light beam. Default: 100")
    parser.add_argument("--refine", type=int, nargs="?", default=0, help="The number of spatial directions will be 12 * 4^refine. Default: 0")
    parser.add_argument("--dist", type=str, nargs="?", default="1", help="The distance to do ray tracing. Dimensionless numbers will be the fraction to the box size. Default: 1. Examples: '1', '1.5_kpc'")
    parser.add_argument("--subsample", type=float, default=0.01, help="Sub-sampling fraction for the star particles. Default: 1.0")
    parser.add_argument("--max_samples", type=int, default=int(1e8), help="The maximum number of samples to do per process. The default is 1e8, which results in 3.2 GB memory usage per process. Increasing this number will increase the speed but also the memory usage linearly.")
    return parser.parse_args()


def process_outputs(args):

    jobdir = args.jobdir
    if args.output is not None:
        outs = [args.output]
    else:
        # Get all the output numbers
        outs = []
        for f in os.listdir(jobdir):
            if f.startswith("output_"):
                try:
                    outs.append(int(f.split("_")[1]))
                except:
                    pass
    if len(outs) == 0:
        sys.exit(f"Error: no output found in {jobdir}")
    outs.sort()
    if len(outs) <= 103:
        print("Found the following outputs:", outs)
    else:
        print("Found the following outputs:", outs[:100], "...", outs[-3:])

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

    for out in outs:
        infopath = f"{jobdir}/output_{out:05d}/info_{out:05d}.txt"
        ds = yt.load(infopath, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        #----- Read star positions and masses  -----
        star_mass = ad["star", "particle_mass"].in_units("Msun")
        star_pos = ad["star", "particle_position"].in_units("code_length")
        star_pos = star_pos.value
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

        t = ds.current_time.in_units("Myr")
        print(f"\nProcessing snapshot {out:05d}, current time: {t:.2f} Myri, number of stars processed: {len(star_mass)}")

        # parse the distance: if it is a number, it is the fraction of the box size, otherwise it is number_unit
        try:
            dist = float(args.dist)
        except:
            assert args.dist.count("_") == 1, "Error: distance must be a number or a number_unit"
            quant = args.dist.replace("_", " ")
            dist = unyt_quantity.from_string(quant) / ds.domain_width[0]
            dist = dist.value
        
        #----- compute column density  -----
        ro = RamsesOutput(jobdir, out)
        sampleNum = args.nsample
        refine = args.refine
        n_directions = 12 * 4**refine
        print("Processing output", out, "with", sampleNum, "samples and", n_directions, "angular directions")
        t0 = time()
        col_1, col_2, col_3 = fesc.col_den_all_stars_and_directions(
            ro, star_pos, n_angular_refine=refine, nsample=sampleNum, H_fraction=0.76, He_fraction=0.24, 
            ray_start=0.0, ray_end=dist, seed=333)
        dt = time() - t0
        print(f"Done processing. Time taken = {dt:.2f} s")

        #----- compute the escape fraction for all directions from all stars  -----
        kappa_HI = fesc.OPACITIES["HI"]
        tau_HI = col_1 * kappa_HI
        fesc_HI = np.exp(-tau_HI)
        kappa_HeI = fesc.OPACITIES["HeI"]
        tau_HeI = col_2 * kappa_HeI
        fesc_HeI = np.exp(-tau_HeI)
        kappa_HeII = fesc.OPACITIES["HeII"]
        tau_HeII = col_3 * kappa_HeII
        fesc_HeII = np.exp(-tau_HeII)
        
        #----- save the results  -----
        outdir = f"{jobdir}/processed/fesc"
        os.makedirs(outdir, exist_ok=True)
        outpath = f"{outdir}/fesc_{out:05d}.npz"
        # save: the escape fraction for all stars and all directions; star masses (Msun); star positions (code_length)
        np.savez(outpath, fesc_HI=fesc_HI, fesc_HeI=fesc_HeI, fesc_HeII=fesc_HeII, star_mass=star_mass, star_pos=star_pos)
    
    return


def compute_fesc(args):

    jobdir = args.jobdir
    if args.output is not None:
        outs = [args.output]
    else:
        # Get all the output numbers
        outs = []
        for f in os.listdir(jobdir):
            if f.startswith("output_"):
                try:
                    outs.append(int(f.split("_")[1]))
                except:
                    pass
    if len(outs) == 0:
        sys.exit(f"Error: no output found in {jobdir}")
    outs.sort()
    if len(outs) <= 103:
        print("\nFound the following outputs:", outs)
    else:
        print("\nFound the following outputs:", outs[:100], "...", outs[-3:])

    for out in outs:
        outpath = f"{jobdir}/processed/fesc/fesc_{out:05d}.npz"
        if not os.path.exists(outpath):
            print(f"Error: {outpath} does not exist. Do data process first. Skipping...")
            continue
        data = np.load(outpath)
        fesc_HI = data["fesc_HI"]
        fesc_HeI = data["fesc_HeI"]
        fesc_HeII = data["fesc_HeII"]
        star_mass = data["star_mass"]
        star_pos = data["star_pos"]

        # print("fesc_HI.shape", fesc_HI.shape)
        # print("fesc_HI.max, min, mean =", fesc_HI.max(), fesc_HI.min(), fesc_HI.mean())
        # return
        
        fesc_HI_star_mean = np.mean(fesc_HI, axis=0)
        fesc_HeI_star_mean = np.mean(fesc_HeI, axis=0)
        fesc_HeII_star_mean = np.mean(fesc_HeII, axis=0)
        fesc_HI_star_and_sky_mean = np.mean(fesc_HI_star_mean)
        fesc_HeI_star_and_sky_mean = np.mean(fesc_HeI_star_mean)
        fesc_HeII_star_and_sky_mean = np.mean(fesc_HeII_star_mean)
        fesc_HI_star_and_sky_std = np.std(fesc_HI_star_mean)
        fesc_HeI_star_and_sky_std = np.std(fesc_HeI_star_mean)
        fesc_HeII_star_and_sky_std = np.std(fesc_HeII_star_mean)
        n_star = fesc_HI.shape[0]
        print(f"Output {out:05d} with {n_star} stars:")
        print("Sky-mean escape fraction for HI, HII, HeII:", fesc_HI_star_and_sky_mean, fesc_HeI_star_and_sky_mean, fesc_HeII_star_and_sky_mean)
        print("Sky-standard deviation for HI, HII, HeII:", fesc_HI_star_and_sky_std, fesc_HeI_star_and_sky_std, fesc_HeII_star_and_sky_std)

        #----- plot the sky map: luminosity weighted escape fraction -----
        # luminosity = fesc.QVacca(star_mass)
        # weights = luminosity * 1e-44
        weights = np.ones(len(star_mass))
        fesc1_sky_weighted = np.dot(weights, fesc_HI) / np.sum(weights)
        fesc.plot_sky(fesc1_sky_weighted, vmin=-6, vmax=0, is_log=True, fn="./sky-cluster")


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
