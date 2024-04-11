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
        print(f"\nProcessing snapshot {out:05d}, current time", ds.current_time.in_units("Myr"))

        #----- Read star positions and masses  -----
        star_mass = ad["star", "particle_mass"].in_units("Msun")
        star_pos = ad["star", "particle_position"].in_units("code_length")
        star_pos = star_pos.value
        n_star = len(star_mass)
        print(f"Found {n_star} stars")
        if n_star == 0:
            continue

        #----- Loading output using Pymses  -----
        ro = RamsesOutput(jobdir, out)
        
        #----- compute column density  -----
        sampleNum = args.nsample
        refine = args.refine
        print("Processing output", out, "with", sampleNum, "samples and", refine, "levels of angular refinement.")
        t0 = time()
        col_1, col_2, col_3 = fesc.col_den_all_stars_and_directions(
            ro, star_pos, n_angular_refine=refine, nsample=sampleNum, H_fraction=0.76, He_fraction=0.24, seed=333)
        dt = time() - t0
        print(f"Time taken = {dt:.2f} s")

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
        np.savez(outpath, fesc_HI=fesc_HI, fesc_HeI=fesc_HeI, fesc_HeII=fesc_HeII)
    
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
        print("Found the following outputs:", outs)
    else:
        print("Found the following outputs:", outs[:100], "...", outs[-3:])

    for out in outs:
        outpath = f"{jobdir}/processed/fesc/fesc_{out:05d}.npz"
        if not os.path.exists(outpath):
            print(f"Error: {outpath} does not exist. Do data process first. Skipping...")
            continue
        data = np.load(outpath)
        fesc_HI = data["fesc_HI"]
        fesc_HeI = data["fesc_HeI"]
        fesc_HeII = data["fesc_HeII"]

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


if __name__ == "__main__":

    Args = arg_parser()
    if Args.task == "process":
        process_outputs(Args)
    elif Args.task == "fesc":
        compute_fesc(Args)
    else:
        sys.exit("Error: task must be either 'process' or 'fesc'")
    print("\nDone.")
