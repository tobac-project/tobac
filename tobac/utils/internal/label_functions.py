"""
Module to allow for labeling of features internally to tobac.
"""

import skimage
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Literal


def label_with_pbcs(
    in_label_arr: np.typing.ArrayLike,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    connectivity: int = 2,
):
    """Function to run labeling, with checks for periodic boundaries.

    Parameters
    ----------
    in_label_arr: ArrayLike (bool)
        Input array to label. Can be 2D or 3D, but needs to be a binary
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        Flag to indicate which boundaries are periodic
    connectivity: int
        What kind of connectivity to use - the sklearn default is 2 (meaning diagonals are included).


    Returns
    -------
    ArrayLike (int)
        A labeled field
    """

    is_3d = len(np.shape(in_label_arr)) == 3

    labels, num_labels = skimage.measure.label(
        in_label_arr, background=0, return_num=True, connectivity=connectivity
    )
    if not is_3d:
        # let's transpose labels to a 1,y,x array to make calculations etc easier.
        labels = labels[np.newaxis, :, :]

    # deal with PBCs
    # all options that involve dealing with periodic boundaries
    pbc_options = ["hdim_1", "hdim_2", "both"]
    if PBC_flag not in pbc_options and PBC_flag != "none":
        raise ValueError(
            "Options for periodic are currently: none, " + ", ".join(pbc_options)
        )

    # we need to deal with PBCs in some way.
    if PBC_flag in pbc_options and num_labels > 0:
        # pairs of labels that touch across a periodic boundary, at the same vertical level
        wall_pairs = list()
        if PBC_flag == "hdim_1" or PBC_flag == "both":
            # north and south wall
            wall_pairs.append((labels[:, 0, :], labels[:, -1, :]))
        if PBC_flag == "hdim_2" or PBC_flag == "both":
            # east/west wall
            wall_pairs.append((labels[:, :, 0], labels[:, :, -1]))
        if PBC_flag == "both":
            # corner points connect to the opposite corner
            wall_pairs.append((labels[:, 0, 0], labels[:, -1, -1]))
            wall_pairs.append((labels[:, 0, -1], labels[:, -1, 0]))

        side_1 = np.concatenate([pair[0].ravel() for pair in wall_pairs])
        side_2 = np.concatenate([pair[1].ravel() for pair in wall_pairs])
        both_labeled = (side_1 != 0) & (side_2 != 0)
        side_1 = side_1[both_labeled]
        side_2 = side_2[both_labeled]

        # merge labels connected across the boundaries (including chains of them),
        # giving each merged label the lowest label number in it
        label_graph = scipy.sparse.coo_matrix(
            (np.ones(len(side_1), dtype=bool), (side_1, side_2)),
            shape=(num_labels + 1, num_labels + 1),
        )
        num_components, component = scipy.sparse.csgraph.connected_components(
            label_graph, directed=False
        )
        lowest_label = np.full(num_components, num_labels + 1, dtype=labels.dtype)
        np.minimum.at(lowest_label, component, np.arange(num_labels + 1))
        new_labels = lowest_label[component]

        # 0 (background) is never connected to anything, so it is its own component
        return new_labels[labels], num_components - 1
    else:
        return labels, num_labels
