"""Code to identify "families" using features from tobac output.

A family is defined as a set of contiguous points above a single threshold that contain at least
one but can contain many detected features.
"""

import pandas as pd
import numpy as np
import xarray as xr
import skimage.measure
import copy
from typing import Optional, Literal, Union
import datetime
import tobac.utils.internal.label_functions as tb_label
import tobac.utils.datetime as tb_datetime


def identify_feature_families_from_segmentation(
    feature_df: pd.DataFrame,
    in_segmentation: xr.DataArray,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    unsegmented_point_values: int = 0,
    below_threshold_values: int = -1,
):
    """
    Function to identify families/storm systems by identifying where segmentation touches.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    in_segmentation: xr.DataArray
        Input segmentation data. Must be derived from the features submitted, but can be
        a subset of those times.
    return_grid: bool
        Whether to return the segmentation grid showing families
    family_column_name: str
        The name in the output dataframe of the family ID
    unsegmented_point_values: int
        The value in the input segmentation for unsegmented but above threshold points
    below_threshold_values: int
        The value in the input segmentation for below threshold points

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    # TODO: This does not currently work if you have segmentation data that
    # uses feature numbers pre-merging

    # we need to label the data, but we currently label using skimage label, not dask label.

    # 3D should be 4-D (time, then 3 spatial).
    # 2D should be 3-D (time, then 2 spatial)
    is_3D = len(in_segmentation.shape) == 4
    seg_family_dict = dict()
    out_families = copy.deepcopy(in_segmentation)
    max_family_number = 0
    enable_family_statistics = True

    if enable_family_statistics:
        region_props_vals = ["bbox", "centroid", "num_pixels"]
        family_stats = dict()

    for time_index in range(in_segmentation.shape[0]):
        in_arr = np.array(in_segmentation.values[time_index])

        segmented_arr = np.logical_and(
            in_arr != unsegmented_point_values, in_arr != below_threshold_values
        )
        # These are our families
        family_labeled_data, number_families = skimage.measure.label(
            segmented_arr, return_num=True
        )
        if enable_family_statistics:
            all_family_nums = list()

            family_props = skimage.measure.regionprops(family_labeled_data)
            for family in family_props:
                all_family_nums.append(family.label + max_family_number)
                family_stats[family.label + max_family_number] = dict()
                # family_stats[family.label+max_family_number]['frame'] =

                family_stats[family.label + max_family_number]["num_pixels"] = family[
                    "num_pixels"
                ]
                family_stats[family.label + max_family_number][family_column_name] = (
                    family.label + max_family_number
                )
                if not is_3D:
                    family_stats[family.label + max_family_number][
                        "hdim_1_center"
                    ] = family["centroid"][0]
                    family_stats[family.label + max_family_number][
                        "hdim_2_center"
                    ] = family["centroid"][1]

                else:
                    # TODO: integrate 3D stats - mostly around center coordinates - need the functions in tobac proper
                    raise NotImplementedError("3D stats not implemented yet")

        # now we need to note feature->family relationship in the dataframe.
        segmentation_props = skimage.measure.regionprops(in_arr)

        # associate feature ID -> family ID
        for seg_area in segmentation_props:
            if is_3D:
                seg_family = family_labeled_data[
                    seg_area.coords[0, 0], seg_area.coords[0, 1], seg_area.cords[0, 2]
                ]
            else:
                seg_family = family_labeled_data[
                    seg_area.coords[0, 0], seg_area.coords[0, 1]
                ]
            seg_family_dict[seg_area.label] = seg_family + max_family_number
        """
        if seg_area is not None and enable_family_statistics:
            curr_feat = seg_area.label
            curr_frame = feature_df[feature_df['feature'] == curr_feat]['frame']
            for family_id in all_family_nums:
                family_stats[family_id]['frame'] = curr_frame.values[0]

        """
        out_families[time_index] = family_labeled_data + max_family_number * (
            (family_labeled_data > unsegmented_point_values).astype(int)
        )
        max_family_number = max_family_number + number_families

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")
    family_df = pd.concat([family_series, feature_series], axis=1)
    out_df = feature_df.merge(family_df, on="feature", how="inner")

    if enable_family_statistics:
        family_stats_df = pd.DataFrame.from_dict(family_stats, orient="index")

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families

    else:
        return out_df, family_stats_df


def identify_feature_families_from_data(
    feature_df: pd.DataFrame,
    in_data: xr.DataArray,
    threshold: float,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    target: Literal["minimum", "maximum"] = "maximum",
    unlinked_family_id: Union[int, None] = -1,
):
    """
    Function to identify families/storm systems by identifying where segmentation touches.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    in_data: xr.DataArray
        Input data. Should match the data that feature_df was generated from.
    threshold: float
        Threshold to define your feature family at
    return_grid: bool
        Whether to return the segmentation grid showing families
    family_column_name: str
        The name in the output dataframe of the family ID
    time_padding: datetime.timedelta
        Time padding to find the matching time between the feature_df and the in_data.
        By default, this is a half second to deal with random errors around time data type
        conversions.
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        What axes to do periodic boundaries on
    target: {"minimum", "maximum"}
        Whether we are looking for things ascending ("maximum") or descending ("minimum")
    unlinked_family_id: int or None
        The value to have in the dataframe for any feature that cannot be linked to a family.
        This is unusual (as every feature should link to a family), but this can happen
        if e.g., the feature position is located outside of the feature area above the threshold.
        If "None", these features are dropped from the output.

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    # we need to label the data, but we currently label using skimage label, not dask label.

    time_var_name = "time"

    # 3D should be 4-D (time, then 3 spatial).
    # 2D should be 3-D (time, then 2 spatial)
    is_3D = len(in_data.shape) == 4

    seg_family_dict = dict()
    out_families = copy.deepcopy(in_data)
    out_families = out_families.astype(np.int64)
    out_families.name = "family_grid"
    max_family_number = 0
    enable_family_statistics = True

    if enable_family_statistics:
        region_props_vals = ["bbox", "centroid", "num_pixels"]
        family_stats = dict()
    for time_index in range(in_data.shape[0]):
        # TODO: fix time_var_name for isel?
        # print("time_index: ", time_index)
        in_data_at_time = in_data.isel(time=time_index)
        in_arr = np.array(in_data_at_time)

        # These are our families
        if target == "minimum":
            mask = in_arr < threshold
        elif target == "maximum":
            mask = in_arr > threshold
        else:
            raise ValueError("target must be minimum or maximum")
        family_labeled_data, number_families = tb_label.label_with_pbcs(
            mask, PBC_flag=PBC_flag, connectivity=1
        )
        if not is_3D:
            family_labeled_data = family_labeled_data[0]
        if enable_family_statistics:
            all_family_nums = list()

            family_props = skimage.measure.regionprops(family_labeled_data)
            # print(max_family_number)
            for family in family_props:
                all_family_nums.append(family.label + max_family_number)
                family_stats[family.label + max_family_number] = dict()
                # family_stats[family.label+max_family_number]['frame'] =

                family_stats[family.label + max_family_number][
                    "num_pixels"
                ] = family.area
                family_stats[family.label + max_family_number][family_column_name] = (
                    family.label + max_family_number
                )
                if not is_3D:
                    family_stats[family.label + max_family_number][
                        "hdim_1_center"
                    ] = family.centroid[0]
                    family_stats[family.label + max_family_number][
                        "hdim_2_center"
                    ] = family.centroid[1]

                else:
                    # TODO: integrate 3D stats - mostly around center coordinates - need the functions in tobac proper
                    raise NotImplementedError("3D stats not implemented yet")

        # need to associate family ID with each feature ID

        # get rows at current time

        rows_at_time = tb_datetime.find_df_rows_at_time(
            feature_df,
            in_data_at_time["time"].values,
            time_var_name=time_var_name,
            time_padding=time_padding,
        )
        rows_at_time = rows_at_time.copy()

        if is_3D:
            v_max, h1_max, h2_max = family_labeled_data.shape
        else:
            # print(family_labeled_data.shape)
            h1_max, h2_max = family_labeled_data.shape

        rows_at_time["hdim_1_adj"] = np.clip(
            (rows_at_time["hdim_1"] + 0.5).astype(int), a_min=0, a_max=h1_max - 1
        )
        rows_at_time["hdim_2_adj"] = np.clip(
            (rows_at_time["hdim_2"] + 0.5).astype(int), a_min=0, a_max=h2_max - 1
        )

        data_in_shape = family_labeled_data.shape
        # TODO: deal with dim order for 3D
        if is_3D:
            if "vdim" not in rows_at_time:
                raise NotImplementedError(
                    "Family ID from raw field not supported going from 2D features to 3D family"
                )

            rows_at_time["vdim_adj"] = np.clip(
                int(rows_at_time["vdim"] + 0.5).astype(int), a_min=0, a_max=v_max - 1
            )
            points_list = (
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
                rows_at_time["vdim_adj"].values,
            )

        else:
            points_list = (
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
            )

        # print(family_labeled_data.shape)

        family_ids = family_labeled_data[points_list]
        # remove 0 (background) if needed

        family_ids_sorted = np.unique(np.sort(family_ids))

        # we want to get rid of points that aren't features in the grid output
        suppressing_families = np.isin(family_labeled_data, family_ids_sorted)
        feature_id_family_id_match_ct = {
            feat: fam_id
            for feat, fam_id in zip(
                rows_at_time["feature"].values, family_ids + max_family_number
            )
        }
        seg_family_dict.update(feature_id_family_id_match_ct)
        out_families[time_index] = (
            family_labeled_data + max_family_number
        ) * suppressing_families.astype(int)

        max_family_number = out_families.max().values

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")
    family_df = pd.concat([family_series, feature_series], axis=1)
    out_df = feature_df.merge(family_df, on="feature", how="inner")
    if unlinked_family_id is not None:
        out_df.loc[out_df["feature_family_id"] == 0, "feature_family_id"] = -1
    else:
        out_df = out_df[out_df["feature_family_id"] != 0]

    if enable_family_statistics:
        family_stats_df = pd.DataFrame.from_dict(family_stats, orient="index")
        fam_to_time_df = out_df[["time", family_column_name]].set_index(
            family_column_name
        )

        fam_to_time_df = fam_to_time_df.loc[
            ~fam_to_time_df.index.duplicated(keep="first"), :
        ].sort_index()
        family_stats_df = family_stats_df.join(
            fam_to_time_df, on=family_column_name
        ).dropna(subset="time")
        family_stats_df = family_stats_df.drop(
            [
                0,
            ],
            axis=0,
            errors="ignore",
        )
        family_stats_df = family_stats_df.set_index(family_column_name)

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families

    else:
        return out_df, family_stats_df
