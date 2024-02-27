from .io import get_hagn_brickfile_stpids, get_pid_HAGN_stars


def gid_to_stars(gid, snap, sim, fields):

    fname = f"/data40b/Horizon-AGN/TREE_STARS/tree_bricks{snap:03d}"

    print("Looking up particle ids")

    pos, rgal, stpids = get_hagn_brickfile_stpids(fname, gid, sim)

    print(f"Fetching particle info from {len(stpids)} particles")

    return get_pid_HAGN_stars(snap, stpids, fields, sim, pos=pos, rgal=rgal)
