"""
Module to allow for labeling of features internally to tobac.
"""

import skimage
import numpy as np
from typing import Literal
import copy
from tobac.utils.internal import label_props
from tobac.utils import periodic_boundaries as pbc_utils


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
    # these are [min, max], meaning that the max value is inclusive and a valid
    # value.
    z_min = 0
    z_max = labels.shape[0] - 1
    y_min = 0
    y_max = labels.shape[1] - 1
    x_min = 0
    x_max = labels.shape[2] - 1

    # deal with PBCs
    # all options that involve dealing with periodic boundaries
    pbc_options = ["hdim_1", "hdim_2", "both"]
    if PBC_flag not in pbc_options and PBC_flag != "none":
        raise ValueError(
            "Options for periodic are currently: none, " + ", ".join(pbc_options)
        )

    # we need to deal with PBCs in some way.
    if PBC_flag in pbc_options and num_labels > 0:
        #
        # create our copy of `labels` to edit
        labels_2 = copy.deepcopy(labels)
        # points we've already edited
        skip_list = np.array([])
        # labels that touch the PBC walls
        wall_labels = np.array([], dtype=np.int32)

        all_label_props = label_props.get_label_props_in_dict(labels)
        [
            all_labels_max_size,
            all_label_locs_v,
            all_label_locs_h1,
            all_label_locs_h2,
        ] = label_props.get_indices_of_labels_from_reg_prop_dict(all_label_props)

        # find the points along the boundaries

        # along hdim_1 or both horizontal boundaries
        if PBC_flag == "hdim_1" or PBC_flag == "both":
            # north and south wall
            ns_wall = np.unique(labels[:, (y_min, y_max), :])
            wall_labels = np.append(wall_labels, ns_wall)

        # along hdim_2 or both horizontal boundaries
        if PBC_flag == "hdim_2" or PBC_flag == "both":
            # east/west wall
            ew_wall = np.unique(labels[:, :, (x_min, x_max)])
            wall_labels = np.append(wall_labels, ew_wall)

        wall_labels = np.unique(wall_labels)

        for label_ind in wall_labels:
            new_label_ind = label_ind
            # 0 isn't a real index
            if label_ind == 0:
                continue
            # skip this label if we have already dealt with it.
            if np.any(label_ind == skip_list):
                continue

            # create list for skip labels for this wall label only
            skip_list_thisind = list()

            # get all locations of this label.
            label_locs_v = all_label_locs_v[label_ind]
            label_locs_h1 = all_label_locs_h1[label_ind]
            label_locs_h2 = all_label_locs_h2[label_ind]

            # loop through every point in the label
            for label_z, label_y, label_x in zip(
                label_locs_v, label_locs_h1, label_locs_h2
            ):
                # check if this is the special case of being a corner point.
                # if it's doubly periodic AND on both x and y boundaries, it's a corner point
                # and we have to look at the other corner.
                # here, we will only look at the corner point and let the below deal with x/y only.
                if PBC_flag == "both" and (
                    np.any(label_y == [y_min, y_max])
                    and np.any(label_x == [x_min, x_max])
                ):
                    # adjust x and y points to the other side
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    label_on_corner = labels[label_z, y_val_alt, x_val_alt]

                    if (label_on_corner != 0) and (
                        ~np.any(label_on_corner == skip_list)
                    ):
                        # alt_inds = np.where(labels==alt_label_3)
                        # get a list of indices where the label on the corner is so we can switch
                        # them in the new list.

                        labels_2[
                            all_label_locs_v[label_on_corner],
                            all_label_locs_h1[label_on_corner],
                            all_label_locs_h2[label_on_corner],
                        ] = label_ind
                        skip_list = np.append(skip_list, label_on_corner)
                        skip_list_thisind = np.append(
                            skip_list_thisind, label_on_corner
                        )

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (np.any(label_on_corner == skip_list_thisind))
                    ):
                        # print("skip_list_thisind label - has already been treated this index")
                        continue

                    # if it's labeled and has already been dealt with via a previous label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (~np.any(label_on_corner == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, x_val_alt]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        skip_list = np.append(skip_list, label_ind)
                        break

                # on the hdim1 boundary and periodic on hdim1
                if (PBC_flag == "hdim_1" or PBC_flag == "both") and np.any(
                    label_y == [y_min, y_max]
                ):
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, y_val_alt, label_x]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        # print(all_label_locs_v[label_alt], alt_inds[0])
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, label_x]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

                if (PBC_flag == "hdim_2" or PBC_flag == "both") and (
                    np.any(label_x == x_min) or np.any(label_x == x_max)
                ):
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, label_y, x_val_alt]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, label_y, x_val_alt]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

        # remove skipped labels from the number, remove 0 from the unique list.
        num_labels = len(np.unique(labels_2)) - 1
        return labels_2, num_labels
    else:
        return labels, num_labels
