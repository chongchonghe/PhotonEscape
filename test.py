import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymses import RamsesOutput
from pymses.analysis import sample_points
from time import time
from pymses.utils import constants as C
import yt
from yt.funcs import mylog
mylog.setLevel(40)

from fesc import fesc

DEBUG = 0
fesc.DEBUG = DEBUG


def test_and_benchmark():

    Aquarius_dir = "./data/Aquarius"
    if not os.path.exists(Aquarius_dir):
        os.makedirs(Aquarius_dir, exist_ok=True)
        print("Aquarius data not found. Do you want to download it (900MB)? (y/n)")
        if input().lower() == "y":
            os.system("wget http://irfu.cea.fr/Projets/coast_documents/Aquarius.tar.gz -O ./data/Aquarius.tar.gz")
            os.system("tar -xzf ./data/Aquarius.tar.gz -C ./data")
    jobdir = os.path.join(Aquarius_dir, "output")
    out = 193

    # jobdir = "data/cluster"
    # out = 273
    # if not os.path.exists(f"{jobdir}/output_{out:05d}"):
    #     sys.exit(f"Error: {jobdir}/output_{out:05d} does not exist. Please place the data there first (or symbolic link to the data folder)")

    try:
        ro = RamsesOutput(jobdir, out)
    except ValueError:
        print("myError: can't find the output folder:")
        print(f"{jobdir}/output_{out:05d}")
        raise SystemExit

    N = 9
    source = ro.amr_source(["rho", "xHII", "xHeII", "xHeIII"])
    dots = np.ones((N, 3)) * 0.5
    dots[:, 2] = np.linspace(0.1, 0.9, N, endpoint=True)

    sp = sample_points(source, dots)

    rho = sp.fields['rho']
    xHII = sp.fields['xHII']
    xHeII = sp.fields['xHeII']
    xHeIII = sp.fields['xHeIII']

    print("rho =")
    print(rho)
    print("xHII =")
    print(xHII)

    t1 = time()
    sp = sample_points(source, dots)
    print(f"N = {N}, Time taken: ", time() - t1, "s")

    N = 1000
    dots = np.ones((N, 3)) * 0.5
    dots[:, 2] = np.linspace(0.1, 0.9, N, endpoint=True)
    t1 = time()
    sp = sample_points(source, dots)
    print(f"N = {N}, Time taken: ", time() - t1, "s")

    N = 10000000
    dots = np.ones((N, 3)) * 0.5
    dots[:, 2] = np.linspace(0.1, 0.9, N, endpoint=True)
    t1 = time()
    sp = sample_points(source, dots)
    print(f"N = {N:.1e}, linear sequential sample, Time taken: ", time() - t1, "s")

    N = 10000000
    dots = np.random.random((N, 3))
    t1 = time()
    sp = sample_points(source, dots)
    print(f"N = {N:.1e}, random sample, Time taken: ", time() - t1, "s")

    print("Test passed.")

    return 


def test_chongchong():

    JOBDIR = "/Users/cche/Documents/time-machine-ignored/Academic/Projects/2020-RAMSES/Job2.2.2"
    # OUT = 15
    OUT = 30

    if not os.path.exists(JOBDIR):
        sys.exit(f"Error: {JOBDIR} does not exist. This test is designed for ChongChong running on his own computer.")

    jobdir = JOBDIR
    out = OUT
    ro = RamsesOutput(jobdir, out)

    #-----  read the sink data  -----
    # f = open("{}/output_{:05d}/sink_{:05d}.info".format(jobdir, out, out))
    # lines = f.readlines()[4:-1]
    # boxlen = np.double(ro.info['boxlen'])
    # star_mass = []
    # star_pos = []
    # for line in lines:
    #     star_mass.append(line.split()[1])
    #     star_pos.append(line.split()[3:6])
    # unit_mass = 1.0 # the mass unit in sink_xxxxx.info is Msun
    # star_mass = np.array(np.double(star_mass)) * unit_mass
    # stars = np.array(np.double(star_pos)) / boxlen
    # # stars = np.array([[0.5, 0.5, 0.5]]) # trivial test

    #-----  read the sink data, NEW  -----
    Msun = C.Msun.express(C.g)
    boxlen = np.double(ro.info['boxlen'])
    unit_density = ro.info['unit_density'].express(C.g / C.cm**3)   # g/cm^3
    # unit_mass = ro.info['unit_mass'].express(C.Msun)                # this is wrong. Damn Pymses!
    unit_length = ro.info['unit_length'].express(C.cm) / boxlen     # cm
    unit_mass_in_Msun = unit_density * unit_length**3 / Msun
    unit_time_in_Myr = ro.info['unit_time'].express(C.Myr)

    sinks = np.loadtxt("{}/output_{:05d}/sink_{:05d}.csv".format(jobdir, out, out), delimiter=",")
    star_pos = sinks[:, 2:5]
    stars = star_pos / boxlen   # star positions normalised to [0, 1]
    star_mass = sinks[:, 1] * unit_mass_in_Msun
    # In He2019, we assume m_star = 0.4 m_sink
    star_mass *= 0.4
    age = sinks[:, -3] * unit_time_in_Myr
    # sanity check, age < 30 Myr
    assert np.all(age < 30), f"Some stars have age > 30 Myr, something must be wrong. The first 10 ages are {age[:10]}"
    lifetimes = fesc.mass_to_lifetime(star_mass)
    is_alive = age < lifetimes
    assert np.sum(is_alive) > 0, "All stars are dead, something must be wrong."

    # test
    if DEBUG > 0:
        stars = np.ones((10, 3)) * 0.9
        star_mass = np.ones(10) * 10.0
        is_alive = np.ones(10, dtype=bool)

    #-----  compute column density  -----
    sampleNum = 100
    nsidePow = 2
    col_1, col_2, col_3 = fesc.col_den_all_stars_and_directions(ro, stars, nsidePow=nsidePow, nsample=sampleNum,
                                                                H_fraction=0.76, He_fraction=0.24, seed=333)
    col1 = np.mean(col_1, axis=0) # average over stars
    print("\nThe column density, averaged over stars, has the following min, max, and mean values (g cm^-2) across the space:")
    print(col1.min(), col1.max(), col1.mean())
    print("and the following shapes:", col1.shape)

    #----- compute the escape fraction for all directions from all stars  -----
    sigma = 6.304e-18
    mH = 1.6733e-24
    kappa = sigma / mH  # cm^2/g
    tau1 = col_1 * kappa
    print("tau has the following min, max, and mean values:")
    print(tau1.min(), tau1.max(), tau1.mean())
    fesc1 = np.exp(-tau1)
    log_fesc1 = - tau1 / np.log(10)
    print("log10(fesc) has the following min, max, and mean values:")
    print(log_fesc1.min(), log_fesc1.max(), log_fesc1.mean())

    #----- Computed luminosity weighted escape fraction  -----
    # mass unit
    luminosity = fesc.QVacca(star_mass)
    # scale down the luminosity by 1e-44 to avoid overflow
    weights = luminosity * 1e-44
    weights[~is_alive] = 0.0
    fesc_weighted = fesc.compute_weighted_fesc(tau1, weights)
    print(f"The luminosity weighted escape fraction is {fesc_weighted}")

    #----- plot the sky map  -----
    fesc1_weighted = np.dot(weights, fesc1) / np.sum(weights)
    fesc.plot_sky(fesc1_weighted, vmin=-2, vmax=0, is_log=True, fn="./sky-chongchong")

    return


def test_cluster():

    jobdir = "data/cluster"
    out = 273

    if not os.path.exists(jobdir):
        sys.exit(f"Error: {jobdir} does not exist. This test is designed for ChongChong running on his own computer.")

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
    f1 = f"{jobdir}/output_{out:05d}/info_{out:05d}.txt"
    # f2 = "output_00274/info_00274.txt"
    ds = yt.load(f1, fields=cell_fields, extra_particle_fields=epf)
    ad = ds.all_data()
    print("snapshot 00273 current time", ds.current_time.in_units("Myr"))

    # pre_rhomax, pre_xyz = ds.find_max(("gas", "density"))
    # print("about to form star cluster at", pre_xyz.to("pc"), "with density", pre_rhomax)
    # stars = np.array([(pre_xyz / ds.length_unit).value])
    # print("star locations =", stars)

    star_mass = ad["star", "particle_mass"].in_units("Msun")
    star_pos = ad["star", "particle_position"].in_units("code_length")
    star_pos = star_pos.value
    # print(star_mass)
    # print(star_pos)
    if DEBUG > 0:
        star_pos = np.ones((10, 3)) * 0.9
        star_mass = np.ones(10) * 10.0
        is_alive = np.ones(10, dtype=bool)

    #----- Loading output using Pymses  -----
    ro = RamsesOutput(jobdir, out)

    #-----  compute column density  -----
    sampleNum = 100
    nsidePow = 0
    col_1, col_2, col_3 = fesc.col_den_all_stars_and_directions(
        ro, star_pos, nsidePow=nsidePow, nsample=sampleNum, H_fraction=0.76, He_fraction=0.24, seed=333)
    col1 = np.mean(col_1, axis=0) # average over stars
    print("\nThe column density, averaged over stars, has the following min, max, and mean values (g cm^-2) across the space:")
    print(col1.min(), col1.max(), col1.mean())
    print("and the following shapes:", col1.shape)

    #----- compute the escape fraction for all directions from all stars  -----
    sigma = 6.304e-18
    mH = 1.6733e-24
    kappa = sigma / mH  # cm^2/g
    tau1 = col_1 * kappa
    print("tau has the following min, max, and mean values:")
    print(tau1.min(), tau1.max(), tau1.mean())
    fesc1 = np.exp(-tau1)
    log_fesc1 = - tau1 / np.log(10)
    print("log10(fesc) has the following min, max, and mean values:")
    print(log_fesc1.min(), log_fesc1.max(), log_fesc1.mean())

    #----- Computed luminosity weighted escape fraction  -----
    # luminosity = fesc.QVacca(star_mass)
    # weights = luminosity * 1e-44
    weights = np.ones(star_pos.shape[0])
    fesc_weighted = fesc.compute_weighted_fesc(tau1, weights)
    print(f"The luminosity weighted escape fraction is {fesc_weighted}")

    #----- plot the sky map  -----
    fesc1_sky_weighted = np.dot(weights, fesc1) / np.sum(weights)
    fesc.plot_sky(fesc1_sky_weighted, vmin=-2, vmax=0, is_log=True, fn="./sky-cluster")

    return


if __name__ == "__main__":
    
    test_and_benchmark()
    # test_chongchong()
    # test_cluster()
