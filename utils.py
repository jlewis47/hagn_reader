from gremlin.read_sim_params import read_info_file, ramses_sim
from zoom_analysis.halo_maker.read_treebricks import convert_star_time

import os
import numpy as np


def get_hagn_sim():

    return ramses_sim(
        "/data52/Horizon-AGN",
        nml="things_for_restart/cosmo.nml",
        info_path="INFO",
        output_path="OUTPUT_DIR",
    )


def convert_hagn_star_units(stars: dict, snap, sim: ramses_sim):

    # sim = get_hagn_sim()

    # info = read_hagn_info(snap)

    # aexp = info["aexp"]

    # unit_l = info["unit_l"]
    # unit_d = info["unit_d"]

    sim.get_snap_exps(param_save=False)
    aexp = sim.aexps[sim.snap_numbers == snap]

    assert len(aexp) == 1, f"couldn't find requested snap: {snap:d} in list:" + str(
        sim.snap_numbers
    )

    # print(aexp, sim.snap_numbers, snap)

    unit_d = sim.cosmo["unit_d"]
    unit_l = sim.cosmo["unit_l"]

    unit_m = unit_d * unit_l**3 / 2e33  # msun

    stars["mass"] *= unit_m

    fried_path = os.path.join(
        os.path.dirname(__file__), "freidmann/"
    )  # create friedmann files
    # in module's directory
    if not os.path.isdir(fried_path):
        os.makedirs(fried_path, exist_ok=True)

    stars["age"] = convert_star_time(
        stars["birth_time"],
        sim,
        aexp,
        cosmo_fname=os.path.join(fried_path, str(snap) + ".txt"),
    )

    del stars["birth_time"]


def hagn_z_to_snap(z):

    # get the redshifts and snapshots
    d = "/data44/Horizon-AGN/INFO"

    info_files = os.listdir(d)

    snaps = np.empty(len(info_files), dtype=int)
    aexps = np.empty(len(info_files), dtype=float)

    for ifile, info_file in enumerate(info_files):
        snaps[ifile] = int(info_file.split("_")[-1].split(".")[0])
        info = read_info_file(os.path.join(d, info_file))
        aexps[ifile] = info["aexp"]

    # print(list(zip(snaps[snaps < 200], 1./aexps[snaps < 200]-1)))

    return snaps[np.argmin(np.abs(1.0 / aexps - 1 - z))]


def adaptahop_to_code_units(x, aexp, sim: ramses_sim):

    box_len = sim.cosmo["unit_l"] / 3.086e24 / sim.aexp_stt * aexp

    return x / box_len + 0.5
