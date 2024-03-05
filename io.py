from re import I
from f90_tools.IO import read_record, read_tgt_fields
from zoom_analysis.halo_maker.read_treebricks import convert_brick_units
from .utils import convert_hagn_star_units, adaptahop_to_code_units

# from zoom_analysis.halo_maker.read_treebricks import convert_brick_units
from gremlin.read_sim_params import ramses_sim

import os
import numpy as np
from scipy.spatial import cKDTree


def read_hagn_star(fname: str, tgt_pos=None, tgt_r=None, tgt_fields: list = None):

    names = [
        "pos",
        "vel",
        "mass",
        "ids",
        "lvl",
        "birth_time",
        "metallicity",
        "mass_init",
    ]

    ndims = [3, 3, 1, 1, 1, 1, 1, 1]

    dtypes = [
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("i4"),
        np.dtype("i4"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
    ]

    steps = np.arange(0, len(names))

    if tgt_fields is None:
        tgt_fields = names

    out = {}

    # find order of tgt_fields in names
    tgt_field_order = [idx for idx, field in zip(steps, names) if field in tgt_fields]
    if len(tgt_field_order) < len(tgt_fields):
        print("Not all fields match requested fields")
        print("Available fields are: ", names)
        print("Requested fields are: ", tgt_fields)
        return out

    tgt_fields = [names[i] for i in tgt_field_order]

    with open(fname, "rb") as f:

        ncpu = read_record(f, 1, "i4")
        # ncpu = np.fromfile(f, dtype="i4", count=3)[1]
        ndim = read_record(f, 1, "i4")
        nstars = read_record(f, 1, "i4")

        # print(ncpu, ndim, nstars)

        tot_nstars = read_record(f, 1, "i4")

        # pos = read_record(f, int(nstars * ndim), "f8").reshape(ndim, -1)
        pos = read_record(f, int(nstars * ndim), "f8").reshape(ndim, -1)

        # print(pos.mean(axis=0))

        pos_filt = False
        if tgt_pos is not None and tgt_r is not None:

            xmax, xmin = pos[0].max(), pos[0].min()
            cond = xmax >= tgt_pos[0] - tgt_r and xmin <= tgt_pos[0] + tgt_r
            if not cond:
                return out
            ymax, ymin = pos[1].max(), pos[1].min()
            cond *= ymax >= tgt_pos[1] - tgt_r and ymin <= tgt_pos[1] + tgt_r
            if not cond:
                return out
            zmax, zmin = pos[2].max(), pos[2].min()
            cond *= zmax >= tgt_pos[2] - tgt_r and zmin <= tgt_pos[2] + tgt_r
            if not cond:
                return out

            # print(cond, xmax, xmin, ymax, ymin, zmax, zmin)

            pos_tree = cKDTree(pos.T, boxsize=1.0 + 1e-10)

            tgt_in_file = pos_tree.query_ball_point(tgt_pos, tgt_r)

            if len(tgt_in_file) == 0:
                return out

            pos = pos[:, tgt_in_file].T

        if "pos" in tgt_fields:
            out["pos"] = pos

        read_tgt_fields(
            out,
            tgt_fields,
            list(zip(names[1:], ndims[1:], dtypes[1:])),
            f,
            nstars,
            args=tgt_in_file,
            debug=False,
        )

        return out


# def get_tgt_HAGN_stars(snap, gid, pties: dict, fields=None):
#     """
#     for a given redshift and galaxy id, get the stars within the galaxy
#     return a dict containing all stellar properties OR only the fields requested

#     pties is a dict containing the galaxy properties
#     """

#     star_dir = f"/data52/Horizon-AGN/STAR/output_star_{snap:05d}"
#     files = os.listdir(star_dir)

#     x = pties["x"][pties["gal_ids"] == gid]
#     y = pties["y"][pties["gal_ids"] == gid]
#     z = pties["z"][pties["gal_ids"] == gid]
#     rgal = pties["rgals"][pties["gal_ids"] == gid]

#     found_stars = {}

#     for ifile, file in enumerate(files):
#         out = read_hagn_star(
#             os.path.join(star_dir, file),
#             tgt_pos=[x, y, z],
#             tgt_r=rgal,
#             tgt_fields=fields,
#         )
#         if len(out) > 0:
#             for k in out.keys():
#                 if k not in found_stars:
#                     found_stars[k] = out[k]
#                 else:
#                     found_stars[k] = np.concatenate([found_stars[k], out[k]])

#     convert_hagn_star_units(found_stars, snap=snap)

#     return found_stars


def get_pid_HAGN_stars(snap, pids, fields, sim: ramses_sim, pos=None, rgal=None):
    """
    for a list of stellar particle ids
    return a dict containing all stellar properties OR only the fields requested


    pties is a dict containing the galaxy properties
    can give pos or rgal to speed up checks
    """
    star_dir = f"/data52/Horizon-AGN/STAR/output_star_{snap:05d}"
    files = os.listdir(star_dir)

    if pos is None and rgal is None:
        print("no pos or rgal given, will check all stars in all files -- this is slow")

    assert (pos is not None and rgal is not None) or (
        pos is None and rgal is None
    ), "must give both pos and rgal or neither"

    found_stars = {}
    npart = len(pids)
    nfound = 0

    for field in fields:
        found_stars[field] = np.empty(npart, "f4")

    if "ids" not in fields:
        fields.append("ids")

    for ifile, file in enumerate(files):
        # print(ifile, nfound)
        fpath = os.path.join(star_dir, file)

        out = read_hagn_star(fpath, tgt_pos=pos, tgt_r=rgal, tgt_fields=fields)

        # print(out.keys())

        if len(out) == 0 or len(out["ids"]) == 0:
            continue

        out["ids"] = np.abs(out["ids"])

        matches = np.isin(out["ids"], pids)
        # print(matches.sum(), out["ids"], pids)
        cur_found = matches.sum()

        # print(cur_found, nfound, npart, len(out["ids"]))
        # print(len(out[field]), len(out[field][matches]))

        if cur_found == 0:
            continue

        for field in found_stars.keys():
            found_stars[field][nfound : nfound + cur_found] = out[field][matches]

        nfound += cur_found

    print(f"Found {nfound:d} stellar particles")

    assert (
        nfound == npart
    ), "didn't find all requested particles... check that position and radius are OK if given"

    convert_hagn_star_units(found_stars, snap, sim)

    return found_stars


def read_hagn_info(snap):

    d = "/data44/Horizon-AGN/INFO"

    fpath = os.path.join(d, f"info_{snap:05d}.txt")

    return np.genfromtxt(fpath, delimiter="=", dtype=None, encoding=None, max_rows=18)


def read_hagn_brickfile(fname, star=False):
    """
    Positions origin is at the center of the box
    Masses (h%m,h%datas%mvir) are in units of 10^11 Msol, and
    Lengths (h%p%x,h%p%y,h%p%z,h%r,h%datas%rvir) are in units of Mpc
    Velocities (h%v%x,h%v%y,h%v%z,h%datas%cvel) are in km/s
    Energies (h%ek,h%ep,h%et) are in
    Temperatures (h%datas%tvir) are in K
    Angular Momentum (h%L%x,h%L%y,h%L%z) are in
    Other quantities are dimensionless (h%my_number,h%my_timestep,h%spin)"""

    if "star" in fname:
        star = True
    elif not star:
        star = False

    # print(star)

    with open(fname, "rb") as src:
        nbodies = read_record(src, 1, np.int32)
        massp = read_record(src, 1, np.float32)
        aexp = read_record(src, 1, np.float32)
        omega_t = read_record(src, 1, np.float32)
        age_univ = read_record(src, 1, np.float32)
        nh, nsub = read_record(src, 2, np.int32)
        nb_structs = nh + nsub
        # print(nbodies, massp, aexp, omega_t, age_univ, nh, nsub, nb_structs)
        # print(nh, nsub, nb_structs)

        # Create empty arrays to store the quantities
        hid = np.empty(nb_structs, dtype=np.int32)
        tstep = np.empty(nb_structs, dtype=np.float32)
        hlvl = np.empty(nb_structs, dtype=np.int32)
        hosth = np.empty(nb_structs, dtype=np.int32)
        hostsub = np.empty(nb_structs, dtype=np.int32)
        nbsub = np.empty(nb_structs, dtype=np.int32)
        nextsub = np.empty(nb_structs, dtype=np.int32)
        hmass = np.empty(nb_structs, dtype=np.float32)
        pos = np.empty((nb_structs, 3), dtype=np.float32)
        vel = np.empty((nb_structs, 3), dtype=np.float32)
        AngMom = np.empty((nb_structs, 3), dtype=np.float32)
        ellipse = np.empty((nb_structs, 4), dtype=np.float32)
        Ek = np.empty(nb_structs, dtype=np.float32)
        Ep = np.empty(nb_structs, dtype=np.float32)
        Et = np.empty(nb_structs, dtype=np.float32)
        spin = np.empty(nb_structs, dtype=np.float32)
        sigma = np.empty(nb_structs, dtype=np.float32)
        sigma_bulge = np.empty(nb_structs, dtype=np.float32)
        m_bulge = np.empty(nb_structs, dtype=np.float32)
        rvir = np.empty(nb_structs, dtype=np.float32)
        mvir = np.empty(nb_structs, dtype=np.float32)
        tvir = np.empty(nb_structs, dtype=np.float32)
        cvel = np.empty(nb_structs, dtype=np.float32)
        rho0 = np.empty(nb_structs, dtype=np.float32)
        r_c = np.empty(nb_structs, dtype=np.float32)

        if star:
            nbin = np.empty(nb_structs, dtype=np.int32)
            rr = np.empty((nb_structs, 100), dtype=np.float32)
            rho = np.empty((nb_structs, 100), dtype=np.float32)

        for istrct in range(nb_structs):
            num_parts = read_record(src, 1, np.int32)

            # Discard particle IDs
            read_record(src, num_parts, np.int32)

            hid[istrct] = read_record(src, 1, np.int32)

            tstep[istrct] = read_record(src, 1, np.float32)

            (
                hlvl[istrct],
                hosth[istrct],
                hostsub[istrct],
                nbsub[istrct],
                nextsub[istrct],
            ) = read_record(src, 5, np.int32)

            hmass[istrct] = read_record(src, 1, np.float32)

            pos[istrct] = read_record(src, 3, np.float32)

            vel[istrct] = read_record(src, 3, np.float32)

            AngMom[istrct] = read_record(src, 3, np.float32)
            ellipse[istrct] = read_record(src, 4, np.float32)
            Ek[istrct], Ep[istrct], Et[istrct] = read_record(src, 3, np.float32)
            spin[istrct] = read_record(src, 1, np.float32)
            sigma[istrct], sigma_bulge[istrct], m_bulge[istrct] = read_record(
                src, 3, np.float32
            )  # skip
            rvir[istrct], mvir[istrct], tvir[istrct], cvel[istrct] = read_record(
                src, 4, np.float32
            )

            rho0[istrct], r_c[istrct] = read_record(src, 2, np.float32)

            if star:
                nbin[istrct] = read_record(src, 1, np.int32)
                rr[istrct, :] = read_record(src, 100, np.float32)
                rho[istrct, :] = read_record(src, 100, np.float32)

        cosmo = {
            "aexp": aexp,
            "omega_t": omega_t,
            "age_univ": age_univ,
        }

        hosting_info = {
            "nh": nh,
            "nsub": nsub,
            "hid": hid,
            "tstep": tstep,
            "hlvl": hlvl,
            "hosth": hosth,
            "hostsub": hostsub,
            "nbsub": nbsub,
            "nextsub": nextsub,
            "hmass": hmass,
        }
        positions = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
        velocities = {"vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2]}
        ellipsis_fit = {
            "r": ellipse[:, 0],
            "a": ellipse[:, 1],
            "b": ellipse[:, 2],
            "c": ellipse[:, 3],
        }
        angular_momentum = {"Lx": AngMom[:, 0], "Ly": AngMom[:, 1], "Lz": AngMom[:, 2]}
        energies = {"Ek": Ek, "Ep": Ep, "Et": Et}
        virial_properties = {
            "spin": spin,
            "rvir": rvir,
            "mvir": mvir,
            "tvir": tvir,
            "cvel": cvel,
        }
        profile_fits = {"rho0": rho0, "r_c": r_c}

        treebrick = {
            "cosmology": cosmo,
            "hosting info": hosting_info,
            "positions": positions,
            "velocities": velocities,
            "angular momentum": angular_momentum,
            "smallest ellipse": ellipsis_fit,
            "energies": energies,
            "virial properties": virial_properties,
            "profile fits": profile_fits,
        }

        return treebrick


def get_hagn_brickfile_stpids(fname, tgt_gid, sim: ramses_sim, star=True):
    """
    returns particle ids associated with tgt galaxy id
    also returns centre and rvir to speed up particle lookup, all other data discarded
    can use for dm brickfiles too, just set star=False and give hid instead of gid
    """

    with open(fname, "rb") as src:
        read_record(src, 1, np.int32)
        read_record(src, 1, np.float32)
        aexp = read_record(src, 1, np.float32)
        read_record(src, 1, np.float32)
        read_record(src, 1, np.float32)
        nh, nsub = read_record(src, 2, np.int32)
        nb_structs = nh + nsub

        for istrct in range(nb_structs):
            num_parts = read_record(src, 1, np.int32)

            pids = read_record(src, num_parts, np.int32)

            gid = read_record(src, 1, np.int32)

            # print(gid)

            if gid == tgt_gid:

                # 3 * 4 * 2 + 7 * 4
                src.seek(52, 1)

                pos = read_record(src, 3, np.float32)
                # print(pos)

                # 6 * 4 * 2 + 17 * 4
                src.seek(116, 1)

                rvir, dummy, dummy, dummy = read_record(src, 4, np.float32)

                pos = adaptahop_to_code_units(pos, aexp, sim)

                return pos, rvir, pids

            else:

                # 3 * 4 * 2 + 7 * 4, 1
                # 1 * 4 * 2 + 3 * 4, 1
                # 6 * 4 * 2 + 17 * 4, 1
                # 1 * 4 * 2 + 4 * 4, 1
                # 1 * 4 * 2 + 2 * 4, 1
                src.seek(228, 1)

                if star:
                    # 3 * 4 * 2 + 201 * 4
                    src.seek(828, 1)

            # read_record(src, 2, np.float32)

    return [], -1, []  # didn't find anything return


def read_hagn_snap_brickfile(snap, sim):

    fname = f"/data40b/Horizon-AGN/TREE_STARS/tree_bricks{snap:03d}"

    brick = read_hagn_brickfile(fname, star=True)

    convert_brick_units(brick, sim)

    return brick
