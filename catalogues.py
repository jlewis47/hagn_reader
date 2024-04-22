import os
from f90_tools.IO import read_record
import numpy as np
import h5py
from gremlin.read_sim_params import ramses_sim

# from hagn.utils import get_hagn_sim


def array_dict(datas):
    d = {}
    for k in datas.dtype.names:
        d[k] = datas[k]

    return d


def read_sfrgal_bin(fname):
    with open(fname, "rb") as f:
        nrow, ncol = read_record(f, 2, np.int32)
        datas = read_record(f, nrow * ncol, dtype=np.float32).reshape(
            (nrow, ncol), order="F"
        )
        # ids, masses, SFR10, SFR 100, SFR1000
        names = ["gid", "mgal", "sfr10", "sfr100", "sfr1000"]
        inds = [0, 5, 1, 2, 3]
        d = {}

        for i, (name, ind) in enumerate(zip(names, inds)):
            d[name] = datas[:, ind]

    return d


def get_galsfr_cat(snap, sim="hagn"):
    assert sim in ["hagn", "nh"]
    if sim == "hagn":
        cat_dir = "/data33/dubois/H-AGN/Catalogs/Gal"
    else:
        cat_dir = "/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/SFR/AdaptaHOP"
    sfrgal_fname = os.path.join(cat_dir, f"list_sfrgal_{snap:05d}.dat")
    return read_sfrgal_bin(sfrgal_fname)


def get_gals_cat(snap, sim="hagn"):
    assert sim in ["hagn", "nh"]
    if sim == "hagn":
        cat_dir = "/data33/dubois/H-AGN/Catalogs/Gal"
        suff = ""
    else:
        cat_dir = "/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/Gal/AdaptaHOP/100PerCentPurity"
        suff = "_nocontam"
    gal_fname = os.path.join(cat_dir, f"list_gal_{snap:05d}.dat" + suff)
    datas = np.genfromtxt(
        gal_fname, names=["gid", "lvl", "mgal", "x", "y", "z", "rgal"]
    )
    return array_dict(datas)


def get_halogals_cat(snap, sim="hagn"):
    assert sim in ["hagn", "nh"]
    if sim == "hagn":
        cat_dir = "/data33/dubois/H-AGN/Catalogs/HaloGal"
        suff = ""
    else:
        cat_dir = "/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/HaloGal/AdaptaHOP/100PerCentPurity"
        suff = "_nocontam"
    halogal_fname = os.path.join(cat_dir, f"halogal_{snap:05d}.dat" + suff)
    datas = np.genfromtxt(halogal_fname, names=["hid", "host", "mhalo", "gid", "mgal"])
    return array_dict(datas)


def get_halos_cat(snap, sim="hagn"):
    assert sim in ["hagn", "nh"]
    if sim == "hagn":
        cat_dir = "/data33/dubois/H-AGN/Catalogs/Halo"
        suff = ""
    else:
        cat_dir = (
            "/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/Halo/100PerCentPurity"
        )
        suff = "_nocontam"

    halo_fname = os.path.join(cat_dir, f"list_halo_{snap:05d}.dat" + suff)

    datas = np.genfromtxt(
        halo_fname, names=["hid", "host", "mhalo", "hx", "hy", "hz", "rvir"]
    )
    return array_dict(datas)


def make_super_cat(snap, sim="hagn", outf=None, overwrite=False):

    assert sim in ["hagn", "nh"]

    fname = f"super_cat_{snap}.h5"

    exists = False
    if outf is not None:
        if not os.path.exists(outf):
            os.makedirs(outf, exist_ok=True)
        exists = os.path.exists(os.path.join(outf, fname))

    if exists and not overwrite:
        print(f"Super catalogue {fname} already exists at given location.")
        # read
        with h5py.File(os.path.join(outf, fname), "r") as f:
            return {key: f[key][...] for key in f.keys()}

    else:

        halos = get_halos_cat(snap, sim=sim)
        halo_gals = get_halogals_cat(snap, sim=sim)
        gals = get_gals_cat(snap, sim=sim)
        galsfrs = get_galsfr_cat(snap, sim=sim)

        # match gal ids in gals and gals_sfrs

        _, sfrgal_args, gal_args = np.intersect1d(
            galsfrs["gid"], gals["gid"], return_indices=True
        )

        super_cat = {}

        for key in galsfrs.keys():
            super_cat[key] = galsfrs[key][sfrgal_args]

        for key in gals.keys():
            if not key in super_cat.keys():
                super_cat[key] = gals[key][gal_args]

        _, super_args, hg_args = np.intersect1d(
            super_cat["gid"], halo_gals["gid"], return_indices=True
        )

        for key in super_cat.keys():
            super_cat[key] = super_cat[key][super_args]

        for key in halo_gals.keys():
            if not key in super_cat.keys():
                super_cat[key] = halo_gals[key][hg_args]

        # match halos and halo_gals

        _, super_args, h_args = np.intersect1d(
            super_cat["hid"], halos["hid"], return_indices=True
        )

        for key in super_cat.keys():
            super_cat[key] = super_cat[key][super_args]

        for key in halos.keys():
            if not key in super_cat.keys():
                super_cat[key] = halos[key][h_args]

        print("super cat keys are: ", super_cat.keys())

        if outf is not None:
            # super_cat = convert_cat_units(super_cat, sim, snap)
            with h5py.File(os.path.join(outf, fname), "w") as f:
                for key in super_cat.keys():
                    f.create_dataset(key, data=super_cat[key])

        return super_cat


def get_cat_gids(cat, gids):
    filt = np.in1d(cat["gid"], gids)
    return {key: cat[key][filt] for key in cat.keys()}


def get_cat_hids(cat, hids):
    filt = np.in1d(cat["hid"], hids)
    return {key: cat[key][filt] for key in cat.keys()}


def convert_cat_units(cat, sim: ramses_sim, snap):

    aexp = sim.get_snap_exps(snap)

    l_sim = sim.cosmo["unit_l"] / 3.08e24 / sim.aexp_stt * aexp  # pMpc

    # print(cat["x"])
    # print(cat["hx"])
    # print(cat["rvir"])

    for pk in ["x", "y", "z", "hx", "hy", "hz", "pos"]:
        cat[pk] += l_sim * 0.5
    for k in ["x", "y", "z", "hx", "hy", "hz", "pos", "rvir", "rgal", "rad", "r"]:
        cat[k] /= l_sim

    if "m" in cat.keys():
        cat["m"] *= 1e11

    return cat


def get_nh_cats_h5(snap, sort="adaptahop"):

    assert sort in ["adaptahop", "hop"]

    if sort == "adaptahop":
        p = os.path.join(f"/data74/cadiou/NewH-catalogs/galaxies-{snap:03d}_dp.h5")
    else:
        p = os.path.join(f"/data74/cadiou/NewH-catalogs/galaxies-{snap:03d}_dp_HOP.h5")

    if not os.path.exists(p):
        print("Catalogue not found for given snapshot.")
        return None

    cat = {}

    with h5py.File(p, "r") as f:

        ks = list(f.keys())

        for k in ks:

            if not hasattr(f[k], "keys"):
                cat[ks] = f[ks]
            else:
                for kk in f[k].keys():
                    cat[kk] = f[k][kk][...]

    return cat
