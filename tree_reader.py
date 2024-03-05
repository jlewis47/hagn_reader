import numpy as np
import os

# from gremlin.read_sim_params import ramses_sim
from f90_tools.IO import read_record, skip_record

# from hagn.utils import get_hagn_sim


def map_tree_rev_steps_bytes(fname, out_path, star=False):

    ##this could be made better by writing to binary... also more storage efficient
    ##will see if I get the time (good luck me!)

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # pass over whole tree and print list of byte numbers that correspond to the start of each step...
    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)
        # print(nsteps)
        # print(read_record(src, 1, np.int32))
        # print(read_record(src, 1, np.int32))
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=False)
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        tree_aexps = read_record(src, nsteps, np.float32)
        # tree_omega_t = read_record(src, nsteps, np.float32)
        # tree_age_univ = read_record(src, nsteps, np.float32)
        skip_record(src, nsteps, np.float32)
        skip_record(src, nsteps, np.float32)

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        byte_positions = np.empty(nsteps, dtype=np.int64)

        for istep in range(nsteps):

            ntot = nb_halos[istep] + nb_shalos[istep]

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
                nbytes[iobj] = src.tell()
                nb_fathers = np.fromfile(src, dtype=np.int32, count=28 + 13 * 2)
                ids[iobj] = nb_fathers[1]
                nb_fathers = nb_fathers[-2]
                # print(nb_fathers)
                if nb_fathers > 0:
                    skip_record(src, nb_fathers, np.float32)
                    skip_record(src, nb_fathers, np.float32)

                nb_sons = read_record(src, 1, np.int32)

                if nb_sons > 0:
                    skip_record(src, nb_sons, np.float32)

                skip_record(src, 1, np.int32)
                skip_record(src, 1, np.int32)
                if star:
                    skip_record(src, 1, np.int32)

            with open(os.path.join(out_path, f"bytes_step_{istep:d}.txt"), "w") as out:
                out.write(f"{byte_positions[istep]}\n")
                for iobj in range(ntot):
                    out.write(f"{ids[iobj]:d},{nbytes[iobj]:d}\n")


def istep_to_nbyte_gal_rev(istep):

    with open(
        os.path.join(
            "/data101/jlewis/hagn/tree_offsets/gal/all_fine_rev",
            f"bytes_step_{istep:d}.txt",
        ),
        "r",
    ) as src:
        line = src.readline()

    return int(line)


def iobj_to_nbyte_gal_rev(istep, obj_id):

    with open(
        os.path.join(
            "/data101/jlewis/hagn/tree_offsets/gal/all_fine_rev",
            f"bytes_step_{istep:d}.txt",
        ),
        "r",
    ) as src:
        for i, line in enumerate(src):
            if i == obj_id:
                break

    return np.int64(line.split(",")[1])


def read_tree_gal_rev(zstart: float, tgt_ids):
    """
    zstart is the redshift at which to start the tree, code will start from closest avaialble step from tree
    tgt_ids is an iterable containing the ids of the objects to follow, starting at z=zstart

    """

    # using the offset files to jump to the right byte position,
    # follow the main branch from zstart to the highest possible redshift
    # in the tree for all tgt_ids

    fname = "/data102/dubois/BigSimsCatalogs/H-AGN/MergerTree/TreeMaker_HAGN_allfinesteps/tree_rev.dat"  # path to the tree

    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)
        # print(nsteps)
        # print(read_record(src, 1, np.int32))
        # print(read_record(src, 1, np.int32))
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=False)
        # nb_halos = np.fromfile(src, dtype=np.int32, count=2 * nsteps + 2)[1:-1]
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        # print(nb_halos, nb_shalos)
        tree_aexps = read_record(src, nsteps, np.float32)
        # tree_omega_t = read_record(src, nsteps, np.float32)
        # tree_age_univ = read_record(src, nsteps, np.float32)
        skip_record(src, nsteps, np.float32)
        skip_record(src, nsteps, np.float32)

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        # skip = tree_aexps > (1.0 / (1.0 + zstart))
        skip = np.argmin(np.abs(tree_aexps - (1.0 / (1.0 + zstart))))

        found_ids = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.int32)
        found_masses = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.float32)

        found_ids[:, 0] = tgt_ids

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        # print(nsteps - skip)

        for istep in range(skip, nsteps):

            nyte_skip = istep_to_nbyte_gal_rev(istep)
            src.seek(nyte_skip)
            # ntot = nb_halos[istep] + nb_shalos[istep]
            # print(f"istep: {istep:d} has {ntot:d} halos+subhalos")
            print(f"step {istep}, z={1.0 / tree_aexps[istep] - 1.0:.4f}")

            # print("byte position is ", src.tell())
            # we aren't at the first step, and didn't find any ids in the previous step... stop the tree
            if istep > skip and np.all(found_ids[:, istep - skip - 1] == -1):
                print("stopping tree... no fathers found in previous step")
                break

            # nfound_this_step = 0

            # for iobj in range(ntot):

            for iobj in np.sort(found_ids[:, istep - skip]):

                obj_bytes = iobj_to_nbyte_gal_rev(istep, iobj)

                src.seek(obj_bytes)

                # print("iobj:", iobj)

                mynumber = read_record(src, 1, np.int32)
                # # print(mynumber)
                # bushID = read_record(src, 1, np.int32)
                # # # print(bushID)
                # mystep = read_record(src, 1, np.int32) - 1  # py indexing
                # # # print(mystep)
                # level, hosthalo, hostsub, nbsub, nextsub = read_record(
                #     src, 5, np.int32, debug=False
                # )

                np.fromfile(src, dtype=np.int32, count=7 + 3 * 2)

                # print(level, hosthalo, hostsub, nbsub, nextsub)
                m = read_record(src, 1, np.float32)
                # print("%e" % (m * 1e11))
                # macc = read_record(src, 1, np.float32)
                # # print(macc)
                # px, py, pz = read_record(src, 3, np.float32)
                # # print(px, py, pz)
                # vx, vy, vz = read_record(src, 3, np.float32)
                # # print(vx, vy, vz)
                # Lx, Ly, Lz = read_record(src, 3, np.float32)
                # # print(Lx, Ly, Lz)
                # r, ra, rb, rc = read_record(src, 4, np.float32)
                # # print(r, ra, rb, rc)
                # ek, ep, et = read_record(src, 3, np.float32)
                # # print(ek, ep, et)
                # spin = read_record(src, 1, np.float32)
                # # print(spin)

                np.fromfile(
                    src, dtype=np.int32, count=18 + 7 * 2
                )  # quickly skip all these entries

                nb_fathers = read_record(src, 1, np.int32)
                # print(nb_fathers)

                if nb_fathers > 0:
                    id_fathers = read_record(src, nb_fathers, np.int32, debug=False)
                    m_fathers = read_record(src, nb_fathers, np.float32)

                    # print(m_fathers)
                    # print(id_fathers.shape, m_fathers.shape)

                    # print(
                    #     mynumber,
                    #     found_ids[:, istep - skip],
                    # )

                    # assert (
                    #     mynumber in found_ids[:, istep - skip]
                    # ), "Error... didn't find expected id at coordinates... byte map file likely wrong or mistmatched"

                    if mynumber in found_ids[:, istep - skip]:

                        found_arg = np.where(found_ids[:, istep - skip] == mynumber)[0]

                        found_masses[found_arg, istep - skip] = m * 1e11

                        if nb_fathers > 1:  # if several follow main branch
                            massive_father = np.argmax(m_fathers)

                            # print(massive_father)

                            main_id = id_fathers[massive_father]
                            # main_m = m_fathers[massive_father]

                            # print(list(zip(id_fathers, m_fathers * 1e11)))
                        else:

                            main_id = id_fathers
                            # main_m = m_fathers
                            # print(id_fathers, m_fathers * 1e11)

                        # print(found_arg, np.sum(found_ids[:, istep - skip] == mynumber))

                        if istep < nsteps - 1:
                            found_ids[found_arg, istep - skip + 1] = main_id
                        # found_masses[found_arg, istep - skip + 1] = main_m * 1e11

                        # print(
                        # mynumber,
                        # found_ids[found_arg, istep - skip + 1],
                        # found_masses[found_arg, istep - skip + 1],
                        # )

                        # nfound_this_step += 1

                        # if nfound_this_step == len(
                        #     tgt_ids
                        # ):  # if I found all the ids at this step... skip to next
                        #     # skip_bytes = zed_to_nbyte(
                        #     #     fbytes, 1.0 / tree_aexps[istep - skip + 1] - 1.0
                        #     # )
                        #     # src.seek(skip_bytes)
                        #     break  # get out of object id

                # since we're skipping exactly to galaxy/halo positions in the tree file, we don't even need to read all of each entry
                # in this case we're going in reverse order so we don't care about the suns at all!
                # nb_sons = read_record(src, 1, np.int32)

                # # print(nb_sons)

                # if nb_sons > 0:
                # id_sons = read_record(src, nb_sons, np.int32)
                # skip_record(src, nb_sons, np.int32)

                #     # print(id_sons)

                # # print(mynumber, id_fathers, id_sons)

                # skip_record(src, 1, np.int32)
                # skip_record(src, 1, np.int32)
                # if star:
                #     skip_record(src, 1, np.int32)

    return found_ids, found_masses, tree_aexps[skip:]


# sim = get_hagn_sim()

# start_z = 2
# end_z = 15

# target_hid = 147479

# snap_nbs = sim.snap_numbers
# snap_aexp = sim.get_snap_exps(param_save=False)
# snap_zeds = 1.0 / snap_aexp - 1.0
# snap_times = sim.get_snap_times(param_save=False)

# start_arg = np.argmin(np.abs(snap_zeds - start_z))
# end_arg = np.argmin(np.abs(snap_zeds - end_z))

# snap_nbs = snap_nbs[start_arg:end_arg]
# snap_aexp = snap_aexp[start_arg:end_arg]
# snap_zeds = snap_zeds[start_arg:end_arg]
# snap_times = snap_times[start_arg:end_arg]

# snap_hids = np.empty(len(snap_nbs), dtype=int)
# snap_masses = np.empty(len(snap_nbs), dtype=float)
# snap_rads = np.empty(len(snap_nbs), dtype=float)


# map_tree_rev_steps_bytes(
#     "/data102/dubois/BigSimsCatalogs/H-AGN/MergerTree/TreeMaker_HAGN_allfinesteps/tree_rev.dat",
#     "/data101/jlewis/hagn/tree_offsets/all_fine_rev/",
# )
