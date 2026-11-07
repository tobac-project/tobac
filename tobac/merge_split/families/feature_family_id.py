"""Code to identify "families" using features from tobac output.

A family is defined as a set of contiguous points above a single threshold that contain at least
one but can contain many detected features.
"""

from typing import Optional, Literal, Union
import datetime

import pandas as pd
import numpy as np
import xarray as xr
import tobac.utils.internal.label_functions as tb_label
import tobac.utils.datetime as tb_datetime


def identify_feature_families_from_segmentation(
    feature_df: pd.DataFrame,
    in_segmentation: xr.DataArray,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
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
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        What axes to do periodic boundaries on

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    booled_values = (in_segmentation != unsegmented_point_values) & (
        in_segmentation != below_threshold_values
    )
    return identify_feature_families_from_data(
        feature_df,
        booled_values,
        threshold=0,
        target="bool",
        return_grid=return_grid,
        PBC_flag=PBC_flag,
        family_column_name=family_column_name,
    )


def identify_feature_families_from_data(
    feature_df: pd.DataFrame,
    in_data: xr.DataArray,
    threshold: float,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    target: Literal["minimum", "maximum", "bool"] = "maximum",
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
    target: {"minimum", "maximum", "bool"}
        Whether we are looking for things ascending ("maximum") or descending ("minimum").
        There is the special case where you already have a true/false array, then you can put
        "bool" as the output.
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
    out_families = xr.zeros_like(in_data, dtype=np.int64)
    out_families.name = "family_grid"
    max_family_number = 0
    enable_family_statistics = False

    if enable_family_statistics:
        family_stats = list()
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
        elif target == "bool":
            mask = in_arr
        else:
            raise ValueError("target must be minimum, maximum, or bool")
        family_labeled_data, number_families = tb_label.label_with_pbcs(
            mask, PBC_flag=PBC_flag, connectivity=1
        )
        if not is_3D:
            family_labeled_data = family_labeled_data[0]

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
                (rows_at_time["vdim"] + 0.5).astype(int), a_min=0, a_max=v_max - 1
            )
            points_list = (
                rows_at_time["vdim_adj"].values,
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
            )

        else:
            points_list = (
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
            )

        # print(family_labeled_data.shape)

        family_ids = family_labeled_data[points_list]
        # remove 0 (background) if needed. Otherwise, a feature that sits on a
        # background point keeps the background in the output grid, offset by
        # max_family_number.

        family_ids_sorted = np.unique(family_ids)
        family_ids_sorted = family_ids_sorted[family_ids_sorted != 0]

        # we want to get rid of points that aren't features in the grid output
        is_kept_family = np.zeros(family_labeled_data.max() + 1, dtype=bool)
        is_kept_family[family_ids_sorted] = True
        suppressing_families = is_kept_family[family_labeled_data]
        feature_id_family_id_match_ct = {
            feat: (fam_id + max_family_number if fam_id != 0 else -1)
            for feat, fam_id in zip(rows_at_time["feature"].values, family_ids)
        }
        seg_family_dict.update(feature_id_family_id_match_ct)
        out_families[time_index] = np.where(
            suppressing_families, family_labeled_data + max_family_number, 0
        )

        # stats are only kept for families with features in them, so only calculate those
        if enable_family_statistics and len(family_ids_sorted) > 0:
            if is_3D:
                # TODO: integrate 3D stats - mostly around center coordinates - need the functions in tobac proper
                raise NotImplementedError("3D stats not implemented yet")
            kept_points = np.flatnonzero(suppressing_families)
            kept_labels = family_labeled_data.ravel()[kept_points]
            kept_h1, kept_h2 = np.unravel_index(kept_points, data_in_shape)
            num_bins = len(is_kept_family)
            num_pixels = np.bincount(kept_labels, minlength=num_bins)
            h1_sum = np.bincount(kept_labels, weights=kept_h1, minlength=num_bins)
            h2_sum = np.bincount(kept_labels, weights=kept_h2, minlength=num_bins)
            num_pixels = num_pixels[family_ids_sorted]
            family_nums = family_ids_sorted.astype(np.int64) + max_family_number
            family_stats.append(
                pd.DataFrame(
                    {
                        "num_pixels": num_pixels,
                        family_column_name: family_nums,
                        "hdim_1_center": h1_sum[family_ids_sorted] / num_pixels,
                        "hdim_2_center": h2_sum[family_ids_sorted] / num_pixels,
                    },
                    index=family_nums,
                )
            )

        # only look at this time; later times still hold the raw input data.
        # keep the running max in case no families were kept at this time.
        max_family_number = max(
            max_family_number, int(out_families[time_index].max().values)
        )

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")
    family_df = pd.concat([family_series, feature_series], axis=1)
    out_df = feature_df.merge(family_df, on="feature", how="inner")
    if unlinked_family_id is not None:
        out_df.loc[out_df[family_column_name] == 0, family_column_name] = -1
    else:
        out_df = out_df[
            np.logical_and(
                out_df[family_column_name] != 0, out_df[family_column_name] != -1
            )
        ]

    if enable_family_statistics:
        if len(family_stats) > 0:
            family_stats_df = pd.concat(family_stats)
        else:
            family_stats_df = pd.DataFrame(
                columns=[
                    "num_pixels",
                    family_column_name,
                    "hdim_1_center",
                    "hdim_2_center",
                ]
            )
        # we need to drop any family_stats that aren't in the feature DF
        family_stats_df = family_stats_df[
            family_stats_df[family_column_name].isin(
                np.unique(family_df[family_column_name].dropna().values)
            )
        ]
        fam_to_time_df = out_df[["time", family_column_name]].set_index(
            family_column_name
        )

        fam_to_time_df = fam_to_time_df.loc[
            ~fam_to_time_df.index.duplicated(keep="first"), :
        ].sort_index()
        fam_to_time_df = fam_to_time_df[fam_to_time_df.index != -1]
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
            return out_df, out_families

    else:
        if enable_family_statistics:
            return out_df, family_stats_df
        else:
            return out_df



def identify_feature_families_from_data_with_mask(
    feature_df: pd.DataFrame,
    feature_mask: xr.DataArray,
    in_data: xr.DataArray,
    threshold: float,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    target: Literal["minimum", "maximum", "bool"] = "maximum",
    unlinked_family_id: Union[int, None] = -1,
    min_overlap_count: Union[int, None] = 64,
):
    """
    Function to identify families/storm systems by identifying where segmentation of in_data intersects at any point with an existing watershed mask feature_mask.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.
    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    feature_mask: xr.DataArray
        Input feature mask. Should match the data that feature_df was generated from.
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
    target: {"minimum", "maximum", "bool"}
        Whether we are looking for things ascending ("maximum") or descending ("minimum").
        There is the special case where you already have a true/false array, then you can put
        "bool" as the output.
    unlinked_family_id: int or None
        The value to have in the dataframe for any feature that cannot be linked to a family.
        This is unusual (as every feature should link to a family), but this can happen
        if e.g., the feature position is located outside of the feature area above the threshold.
        If "None", these features are dropped from the output.
    min_overlap_count: int or None
        The minimum number of overlap points needed between input feature mask and mask from data
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
    seg_overlap_count_dict = dict()
    out_families = copy.deepcopy(in_data)
    out_families = out_families.astype(np.int64)
    out_families.name = "family_grid"
    max_family_number = 0
    enable_family_statistics = False

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
        elif target == "bool":
            mask = in_arr
        else:
            raise ValueError("target must be minimum, maximum, or bool")
        family_labeled_data, number_families = label_with_pbcs(
            mask, PBC_flag=PBC_flag, connectivity=1
        )
        if not is_3D:
            family_labeled_data = family_labeled_data[0]


        # need to associate family ID with each feature ID

        # get rows at current time

        rows_at_time = find_df_rows_at_time(
            feature_df,
            in_data_at_time["time"].values,
            time_var_name=time_var_name,
            time_padding=time_padding,
        )
        rows_at_time = rows_at_time.copy()

        #print(family_labeled_data.shape)

        # instead of extracting the specific updraft center point here, should check for an overlap with an input mask (from watershedding) 
        # then assign each feature a family ID based on which region the W region intersects with the most. 

        feature_mask_at_time = feature_mask.isel(time=time_index).values

        # find overlap of the family mask and original feature mask
        overlap_mask = (family_labeled_data>0) & (feature_mask_at_time>0)

        # get corresponding family and feature IDs for overlap points
        family_ids = family_labeled_data[overlap_mask]
        feature_ids =feature_mask_at_time[overlap_mask]

        # for each overlap point in space, assign the feature ID with associated family ID
        overlap_feature_family_ids = np.column_stack((feature_ids, family_ids))
        overlap_feature_family_ids, counts = np.unique(overlap_feature_family_ids, axis=0, return_counts=True)

        feature_id_family_id_match_ct = {}
        feature_id_overlap_count = {}
        max_counts = {}


        # for each feature ID, find which family ID has the largest overlapping region
        # and assign the feature ID to that family ID
        for (feat_id, fam_id), count in zip(overlap_feature_family_ids, counts):
            if (count > max_counts.get(feat_id,0)) & (count > min_overlap_count):

                max_counts[feat_id] = count
                feature_id_family_id_match_ct[feat_id] = fam_id+max_family_number
                feature_id_overlap_count[feat_id] = count

        seg_family_dict.update(feature_id_family_id_match_ct)
        seg_overlap_count_dict.update(feature_id_overlap_count)

        out_families[time_index] = (
            family_labeled_data + max_family_number
        )

        max_family_number = out_families.max().values

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")

    overlap_count_series = pd.Series(seg_overlap_count_dict, name=f"{family_column_name}_ncells_overlap")

    family_df = pd.concat([family_series, feature_series, overlap_count_series], axis=1)

    # note: as written, this currently drops any features that do not have overlap with any families
    out_df = feature_df.merge(family_df, on="feature", how="inner")

    if unlinked_family_id is not None:
        out_df.loc[out_df[family_column_name] == 0, family_column_name] = -1
    else:
        out_df = out_df[
            np.logical_and(
                out_df[family_column_name] != 0, out_df[family_column_name] != -1
            )
        ]

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families
        else:
            return out_df, out_families

    else:
        if enable_family_statistics:
            return out_df, family_stats_df
        else:
            return out_df
