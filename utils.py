from gremlin.read_sim_params import read_info_file, ramses_sim
from zoom_analysis.halo_maker.read_treebricks import convert_star_time

import os
import numpy as np


def get_hagn_sim():

    sim = ramses_sim(
        "/data52/Horizon-AGN",
        nml="things_for_restart/cosmo.nml",
        info_path="INFO",
        output_path="OUTPUT_DIR",
        sink_path="/data40a/Horizon-AGN/SINK_PROPS",
        param_save=False,
    )

    hydro = {
        "1": "density",
        "2": "velocity_x",
        "3": "velocity_y",
        "4": "velocity_z",
        "5": "pressure",
        "6": "metallicity",
    }

    sim.hydro = hydro

    return sim


def get_nh_sim():

    return ramses_sim(
        "/data7c/NewHorizon",
        nml="Things_For_Restart/cosmo.nml",
        info_path="INFO",
        output_path="OUTPUT_DIR",
        sink_path="/data7c/NewHorizon/SINKPROPS",
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

    if "mass" in stars:
        # unit_d = sim.cosmo["unit_d"]
        # unit_l = sim.cosmo["unit_l"]

        unit_m = sim.unit_d(aexp) * sim.unit_l(aexp) ** 3 / 2e33  # msun

        stars["mass"] *= unit_m

    if "birth_time" in stars:
        # fried_path = os.path.join(
        #     os.path.dirname(__file__), "friedmann/"
        # )  # create friedmann files
        # # in module's directory
        # if not os.path.isdir(fried_path):
        #     os.makedirs(fried_path, exist_ok=True)

        stars["age"] = convert_star_time(
            stars["birth_time"],
            sim,
            aexp,
            # cosmo_fname=os.path.join(fried_path, "friedman" + ".txt"),
        )

        del stars["birth_time"]

    if "ids" in stars:
        stars["ids"] = np.abs(stars["ids"])


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

    box_len = sim.unit_l(aexp) / 3.08e24  # * aexp

    return x / box_len + 0.5
