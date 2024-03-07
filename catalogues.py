import os
from f90_tools.IO import read_record
import numpy as np
import h5py


def array_dict(datas):
    d = {}
    for k in datas.dtype.names:
        d[k] = datas[k]

    return d


gal_dir = "/data33/dubois/H-AGN/Catalogs/Gal"


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


def get_galsfr_cat(snap):
    sfrgal_fname = os.path.join(gal_dir, f"list_sfrgal_{snap:05d}.dat")
    return read_sfrgal_bin(sfrgal_fname)


def get_gals_cat(snap):
    gal_fname = os.path.join(gal_dir, f"list_gal_{snap:05d}.dat")
    datas = np.genfromtxt(
        gal_fname, names=["gid", "lvl", "mgal", "x", "y", "z", "rgal"]
    )
    return array_dict(datas)


def get_halogals_cat(snap):
    halogal_fname = os.path.join(
        gal_dir.replace("Gal", "HaloGal"), f"halogal_{snap:05d}.dat"
    )
    datas = np.genfromtxt(halogal_fname, names=["hid", "host", "mhalo", "gid", "mgal"])
    return array_dict(datas)


def get_halos_cat(snap):
    halo_fname = os.path.join(
        gal_dir.replace("Gal", "Halo"), f"list_halo_{snap:05d}.dat"
    )

    datas = np.genfromtxt(
        halo_fname, names=["hid", "host", "mhalo", "hx", "hy", "hz", "rvir"]
    )
    return array_dict(datas)


def make_super_cat(snap, outf=None, overwrite=False):

    fname = "super_cat_{snap}.h5"

    exists = False
    if outf is not None:
        if not os.path.exists(outf):
            os.makedirs(outf, exist_ok=True)
        exists = os.path.exists(os.path.join(outf, fname))

    if exists and not overwrite:
        print(f"Super catalogue {outf} already exists")
        # read
        with h5py.File(outf, "r") as f:
            return {key: f[key][...] for key in f.keys()}

    else:

        halos = get_halos_cat(snap)
        halo_gals = get_halogals_cat(snap)
        gals = get_gals_cat(snap)
        galsfrs = get_galsfr_cat(snap)

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
            with h5py.File(os.path.join(outf, fname), "w") as f:
                for key in super_cat.keys():
                    f.create_dataset(key, data=super_cat[key])

        return super_cat
