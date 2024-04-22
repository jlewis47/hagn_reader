from importlib.util import spec_from_file_location
import numpy as np
import os

from f90_tools.IO import read_record, skip_record

from hagn.utils import adaptahop_to_code_units, get_hagn_sim
from hagn.io import get_hagn_brickfile_aexp, get_hagn_brickfile_ids
from hagn.association import gid_to_stars

import h5py


# from zoom_analysis.halo_maker.read_treebricks import (
#     convert_brick_units,
# )


def map_tree_steps_bytes(fname, out_path, star=True, sim="hagn", debug=False):

    assert sim in ["hagn", "nh"], "possible sims are 'hagn' or 'nh'"
    # assert direction in ["rev", "fwd"], "possible directions are 'rev' or 'fwd'"

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    float_dtype = np.float32
    # father_skip = (
    #     28  # hardcoded number of bytes depending on number of entries and dtype
    # ) #maybe perf gain... but makes things more complex for little reason.
    if sim == "nh" and ("GAL" in fname or "gal" in fname):
        float_dtype = np.float64
        # father_skip = 47

    print(f"using float dtype: {float_dtype}")  # and father skip: {father_skip}")

    # pass over whole tree and print list of byte numbers that correspond to the start of each step...
    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)
        # print(nsteps)
        # print(read_record(src, 1, np.int32))
        # print(read_record(src, 1, np.int32))
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=debug)
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        tree_aexps = read_record(src, nsteps, float_dtype)
        # tree_omega_t = read_record(src, nsteps, float_dtype)
        # tree_age_univ = read_record(src, nsteps, float_dtype)
        skip_record(src, 1)
        skip_record(src, 1)

        # print(nb_halos, nb_shalos, tree_aexps)  # , tree_omega_t, tree_age_univ)

        byte_positions = np.empty(nsteps, dtype=np.int64)

        for istep in range(nsteps):

            ntot = nb_halos[istep] + nb_shalos[istep]

            #  print(ntot)

            ids = np.empty(ntot, dtype=np.int32)
            nbytes = np.empty(ntot, dtype=np.int64)

            print(f"istep: {istep:d} has {ntot:d} halos+subhalos")
            print(f"redshift was {1.0 / tree_aexps[istep] - 1.0:.4f}")

            byte_pos = src.tell()

            # print("byte position is ", byte_pos)

            byte_positions[istep] = byte_pos

            for iobj in range(ntot):

                # print(iobj)

                # skip_record(src, 27, dtype=np.int32)
                # nb_fathers = read_record(src, 1, dtype=np.int32)
                # nbytes[iobj] = src.tell()
                # nb_fathers = np.fromfile(
                #     src, dtype=np.int32, count=father_skip + 13 * 2
                # )
                # ids[iobj] = nb_fathers[1]
                # print(nb_fathers)
                # print(ids[iobj])
                # nb_fathers = nb_fathers[-2]
                # print(nb_fathers)

                ids[iobj] = read_record(src, 1, np.int32)
                skip_record(src, 11)
                nb_fathers = read_record(src, 1, np.int32)

                # print(ids[iobj], nb_fathers)
                if nb_fathers > 0:
                    skip_record(src, 1)
                    skip_record(src, 1)

                nb_sons = read_record(src, 1, np.int32)
                # print(nb_sons)

                if nb_sons > 0:
                    skip_record(src, 1)

                skip_record(src, 1)
                skip_record(src, 1)
                if star == False:
                    skip_record(src, 1)

            with h5py.File(
                os.path.join(out_path, f"bytes_step_{istep:d}.h5"), "w"
            ) as out:
                # out.write(f"{byte_positions[istep]}")
                out.create_dataset(
                    "step_nbytes",
                    data=byte_positions[istep],
                    dtype=np.int64,
                )
                out.create_dataset(
                    "obj_ids", data=ids, dtype=np.int32, compression="lzf"
                )
                out.create_dataset(
                    "obj_nbytes", data=nbytes, dtype=np.int64, compression="lzf"
                )


def istep_to_nbyte(istep, tree_type="gal", sim="hagn", direction="rev"):

    fpath = f"/data101/jlewis/hagn/tree_offsets/{tree_type}/all_fine_{direction:s}"
    if sim == "nh":
        fpath = f"/data101/jlewis/hn/tree_offsets/{tree_type}/{direction:s}"

    with h5py.File(os.path.join(fpath, f"bytes_step_{istep:d}.h5"), "r") as src:
        return int(src["step_nbytes"][()])


def iobj_to_nbyte(istep, obj_id, tree_type="gal", sim="hagn", direction="rev"):

    fpath = f"/data101/jlewis/hagn/tree_offsets/{tree_type}/all_fine_{direction:s}"
    if sim == "nh":
        fpath = f"/data101/jlewis/hn/tree_offsets/{tree_type}/{direction:s}"

    with h5py.File(os.path.join(fpath, f"bytes_step_{istep:d}.h5"), "r") as src:
        # ids = src["obj_ids"][()]
        # find_line = np.searchsorted(ids, obj_id)
        return src["obj_nbytes"][obj_id - 1]


def read_tree(
    zstart: float,
    tgt_ids,
    tree_type="gal",
    target_fields=None,
    sim="hagn",
    direction="rev",
):
    """
    zstart is the redshift at which to start the tree, code will start from closest avaialble step from tree
    tgt_ids is an iterable containing the ids of the objects to follow, starting at z=zstart

    """

    assert tree_type in ["gal", "halo"], "possible types are 'gal' or 'halo'"

    assert sim in ["hagn", "nh"], "possible sims are 'hagn' or 'nh'"

    # using the offset files to jump to the right byte position,
    # follow the main branch from zstart to the highest possible redshift
    # in the tree for all tgt_ids

    if direction == "fwd":
        tree_name = "tree.dat"
    elif direction == "rev":
        tree_name = "tree_rev.dat"
    else:
        raise ValueError("direction must be 'fwd' or 'rev'")

    float_dtype = np.float32
    if sim == "hagn":
        if tree_type == "gal":
            fname = f"/data102/dubois/BigSimsCatalogs/H-AGN/MergerTree/TreeMaker_HAGN_allfinesteps/{tree_name:s}"  # path to the tree
        else:
            fname = f"/data102/dubois/BigSimsCatalogs/H-AGN/MergerTreeHalo/HAGN/{tree_name:s}"
    elif sim == "nh":
        if tree_type == "gal":
            float_dtype = np.float64
            fname = f"/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/MergerTrees/Halo/{tree_name:s}"  # path to the tree
        else:
            fname = f"/data102/dubois/BigSimsCatalogs/NewHorizon/Catalogs/MergerTrees/Gal/AdaptaHOP/{tree_name:s}"

    if target_fields is None:
        target_fields = ["m"]

    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)

        nb_halos = read_record(src, nsteps * 2, np.int32, debug=False)
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        tree_aexps = read_record(src, nsteps, float_dtype)
        skip_record(src, 1)
        skip_record(src, 1)

        skip = np.argmin(np.abs(tree_aexps - (1.0 / (1.0 + zstart))))

        found_ids = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.int32)

        found_fields = {}
        for tgt_f in target_fields:
            found_fields[tgt_f] = np.full(
                (len(tgt_ids), nsteps - skip), -1, dtype=np.float32
            )

        found_ids[:, 0] = np.sort(tgt_ids)  # earlier ids are earlier in file

        for istep in range(skip, nsteps):

            nyte_skip = istep_to_nbyte(istep, tree_type=tree_type, sim=sim)
            src.seek(
                nyte_skip
            )  # no difference to perf if use byte position from start vs relative position from current position

            # print(f"step {istep}, z={1.0 / tree_aexps[istep] - 1.0:.4f}")

            if istep > skip and np.all(found_ids[:, istep - skip - 1] == -1):
                print("stopping tree... no fathers found in previous step")
                break

            # for iobj in np.sort(found_ids[:, istep - skip]):
            for iobj in found_ids[:, istep - skip]:

                obj_bytes = iobj_to_nbyte(istep, iobj, tree_type=tree_type, sim=sim)

                src.seek(obj_bytes)

                mynumber = read_record(src, 1, np.int32)
                bushID = read_record(src, 1, np.int32)
                mystep = read_record(src, 1, np.int32) - 1  # py indexing
                level, hosthalo, hostsub, nbsub, nextsub = read_record(
                    src, 5, np.int32, debug=False
                )

                # np.fromfile(src, dtype=np.int32, count=7 + 3 * 2)

                m = read_record(src, 1, float_dtype)

                if direction == "fwd" and sim == "hagn" and tree_type == "gal":
                    macc = read_record(
                        src, 1, np.float64
                    )  # for some reason we need this...
                else:
                    macc = read_record(src, 1, float_dtype)
                px, py, pz = read_record(src, 3, float_dtype)
                vx, vy, vz = read_record(src, 3, float_dtype)
                Lx, Ly, Lz = read_record(src, 3, float_dtype)
                r, ra, rb, rc = read_record(src, 4, float_dtype)
                ek, ep, et = read_record(src, 3, float_dtype)
                spin = read_record(src, 1, float_dtype)
                #
                nb_fathers = read_record(src, 1, np.int32)
                # print(nb_fathers)

                if nb_fathers > 0:
                    id_fathers = read_record(src, nb_fathers, np.int32, debug=False)
                    m_fathers = read_record(src, nb_fathers, np.float32)

                    if mynumber in found_ids[:, istep - skip]:

                        found_arg = np.where(found_ids[:, istep - skip] == mynumber)[0]

                        # found_masses[found_arg, istep - skip] = m * 1e11

                        if "m" in target_fields:
                            found_fields["m"][found_arg, istep - skip] = m * 1e11
                        if "level" in target_fields:
                            found_fields["level"][found_arg, istep - skip] = level
                        if "hosthalo" in target_fields:
                            found_fields["hosthalo"][found_arg, istep - skip] = hosthalo
                        if "hostsub" in target_fields:
                            found_fields["hostsub"][found_arg, istep - skip] = hostsub
                        if "nbsub" in target_fields:
                            found_fields["nbsub"][found_arg, istep - skip] = nbsub
                        if "nextsub" in target_fields:
                            found_fields["nextsub"][found_arg, istep - skip] = nextsub
                        if "x" in target_fields:
                            found_fields["x"][found_arg, istep - skip] = px
                        if "y" in target_fields:
                            found_fields["y"][found_arg, istep - skip] = py
                        if "z" in target_fields:
                            found_fields["z"][found_arg, istep - skip] = pz
                        if "vx" in target_fields:
                            found_fields["vx"][found_arg, istep - skip] = vx
                        if "vy" in target_fields:
                            found_fields["vy"][found_arg, istep - skip] = vy
                        if "vz" in target_fields:
                            found_fields["vz"][found_arg, istep - skip] = vz
                        if "Lx" in target_fields:
                            found_fields["Lx"][found_arg, istep - skip] = Lx
                        if "Ly" in target_fields:
                            found_fields["Ly"][found_arg, istep - skip] = Ly
                        if "Lz" in target_fields:
                            found_fields["Lz"][found_arg, istep - skip] = Lz
                        if "r" in target_fields:
                            found_fields["r"][found_arg, istep - skip] = r
                        if "ra" in target_fields:
                            found_fields["ra"][found_arg, istep - skip] = ra
                        if "rb" in target_fields:
                            found_fields["rb"][found_arg, istep - skip] = rb
                        if "rc" in target_fields:
                            found_fields["rc"][found_arg, istep - skip] = rc
                        if "ek" in target_fields:
                            found_fields["ek"][found_arg, istep - skip] = ek
                        if "ep" in target_fields:
                            found_fields["ep"][found_arg, istep - skip] = ep
                        if "et" in target_fields:
                            found_fields["et"][found_arg, istep - skip] = et
                        if "spin" in target_fields:
                            found_fields["spin"][found_arg, istep - skip] = spin

                        if nb_fathers > 1:  # if several follow main branch
                            massive_father = np.argmax(m_fathers)

                            main_id = id_fathers[massive_father]

                        else:

                            main_id = id_fathers

                        if "m_father" in target_fields:
                            found_fields["m_father"][found_arg, istep - skip] = (
                                m_fathers[massive_father]
                            )

                        if istep < nsteps - 1:
                            found_ids[found_arg, istep - skip + 1] = main_id

    return found_ids, found_fields, tree_aexps[skip:]


# map_tree_rev_steps_bytes(
#     "/data102/dubois/BigSimsCatalogs/H-AGN/MergerTree/TreeMaker_HAGN_allfinesteps/tree_rev.dat",
#     "/data101/jlewis/hagn/tree_offsets/all_fine_rev/",
# )


def follow_treebricks(aexp_stt, aexp_end, brick_dir, tree_aexps, ids, fields=None):
    """
    ids needs to be 2D : (n galaxies, n tree steps)
    """

    hagn_sim = get_hagn_sim()

    if fields is None:
        return {}

    brick_files = [f for f in os.listdir(brick_dir) if "tree_bricks" in f]
    brick_fnames = np.asarray([os.path.join(brick_dir, brick) for brick in brick_files])

    brick_numbers = np.asarray([int(brick[-3:]) for brick in brick_files])

    brick_fnames = brick_fnames[np.argsort(brick_numbers)]
    brick_aexps = np.asarray([get_hagn_brickfile_aexp(brick) for brick in brick_fnames])

    in_range = (brick_aexps <= aexp_stt) & (brick_aexps >= aexp_end)

    # print(brick_aexps[in_range])
    # print(1.0 / brick_aexps[in_range] - 1)

    print(f"Found {np.sum(in_range)} bricks in range")

    found_fields = {}
    # brick_aexps = np.zeros(len(brick_files))
    for tgt_f in fields:
        found_fields[tgt_f] = np.full(
            (len(ids), len(brick_aexps)), -1, dtype=np.float32
        )

    for ibrick, brick in enumerate(brick_fnames[in_range]):

        # brick_num = int(brick.split("_")[-1].split(".")[0])

        print(ibrick, brick)

        # closest tree step
        closest_step = np.argmin(np.abs(tree_aexps - brick_aexps[ibrick]))
        # print(np.shape(ids), closest_step)
        closest_ids = ids[:, closest_step]

        brick_data = get_hagn_brickfile_ids(brick, closest_ids, star=True)

        # print(brick_data["hmass"])
        # print(brick_data["pos"])

        # print(closest_ids, brick_data["hosting info"]["hid"])

        # print(np.in1d(closest_ids, brick_data["hosting info"]["hid"]))

        # args = np.where(brick_data["hosting info"]["hid"] == closest_ids[:, None])[1]

        # print(args)

        for f in brick_data.keys():
            if f in fields:
                found_fields[f][:, ibrick] = brick_data[f]
                # print(f, brick_data[f], found_fields[f][:, ibrick])

    box_len = (
        hagn_sim.cosmo["unit_l"] / 3.08e24 / hagn_sim.aexp_stt * brick_aexps[ibrick]
    )  # / h / aexp  # * aexp  # proper Mpc

    if "hmass" in brick_data.keys():
        brick_data["hmass"] *= 1e11
    if "mvir" in brick_data.keys():
        brick_data["mvir"] *= 1e11

    if "pos" in brick_data.keys():
        brick_data["pos"] /= box_len
        brick_data["pos"] += 0.5

    return found_fields, brick_aexps


def follow_treebricks_sfr(aexp_stt, aexp_end, brick_dir, tree_aexps, ids):
    """
    ids needs to be 2D : (n galaxies, n tree steps)
    """

    hagn_sim = get_hagn_sim()

    hagn_aexps = hagn_sim.aexps
    hagn_snaps = hagn_sim.snaps

    brick_files = [f for f in os.listdir(brick_dir) if "tree_bricks" in f]
    brick_fnames = np.asarray([os.path.join(brick_dir, brick) for brick in brick_files])

    brick_numbers = np.asarray([int(brick[-3:]) for brick in brick_files])

    brick_fnames = brick_fnames[np.argsort(brick_numbers)]
    brick_aexps = np.asarray([get_hagn_brickfile_aexp(brick) for brick in brick_fnames])

    in_range = (brick_aexps <= aexp_stt) & (brick_aexps >= aexp_end)

    print(f"Found {np.sum(in_range)} bricks in range")

    mstar = np.zeros((len(ids), in_range.sum()))
    sfr = np.zeros((len(ids), in_range.sum()))

    # brick_aexps = np.zeros(len(brick_files))

    for ibrick, brick, snap in enumerate(
        brick_fnames[in_range], brick_numbers[in_range]
    ):

        # brick_num = int(brick.split("_")[-1].split(".")[0])

        print(ibrick, brick)

        # closest tree step
        closest_step = np.argmin(np.abs(tree_aexps - brick_aexps[ibrick]))
        # print(np.shape(ids), closest_step)
        closest_ids = ids[:, closest_step]

        for i, gid in enumerate(closest_ids):

            print(gid)

            stars = gid_to_stars(gid, snap, hagn_sim, ["mass", "age", "metallicity"])
            brick_data = get_hagn_brickfile_ids(brick, gid, star=True)

            mstar[i, ibrick] = brick_data["mstar"]
            sfr[i, ibrick] = brick_data["sfr"]
