"""Provide essential methods for masking"""

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from tobac.utils.generators import field_and_features_over_time


def column_mask_from2D(mask_2D, cube, z_coord="model_level_number"):
    """Turn 2D watershedding mask into a 3D mask of selected columns.

    Parameters
    ----------
    cube : iris.cube.Cube
        Data cube.

    mask_2D : iris.cube.Cube
        2D cube containing mask (int id for tacked volumes 0
        everywhere else).

    z_coord : str
        Name of the vertical coordinate in the cube.

    Returns
    -------
    mask_2D : iris.cube.Cube
        3D cube containing columns of 2D mask (int id for tracked
        volumes, 0 everywhere else).
    """

    from copy import deepcopy

    mask_3D = deepcopy(cube)
    mask_3D.rename("segmentation_mask")
    dim = mask_3D.coord_dims(z_coord)[0]
    for i in range(len(mask_3D.coord(z_coord).points)):
        slc = [slice(None)] * len(mask_3D.shape)
        slc[dim] = slice(i, i + 1)
        mask_out = mask_3D[slc]
        mask_3D.data[slc] = mask_2D.core_data()
    return mask_3D


def mask_cube_cell(variable_cube, mask, cell, track):
    """Mask cube for tracked volume of an individual cell.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tracked volumes, 0 everywhere
        else).

    cell : int
        Integer id of cell to create masked cube for.

    track : pandas.DataFrame
        Output of the linking.

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube with data for respective cell.
    """

    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    variable_cube_out = mask_cube_features(variable_cube, mask, feature_ids)
    return variable_cube_out


def mask_cube_all(variable_cube, mask):
    """Mask cube (iris.cube) for tracked volume.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube for untracked volume.
    """

    from dask.array import ma
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        mask.core_data() == 0, variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_cube_untracked(variable_cube, mask):
    """Mask cube (iris.cube) for untracked volume.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube for untracked volume.
    """

    from dask.array import ma
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        mask.core_data() > 0, variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_cube(cube_in, mask):
    """Mask cube where mask is not zero.

    Parameters
    ----------
    cube_in : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Mask to use for masking, >0 where cube is supposed to be masked.

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube.
    """

    from dask.array import ma
    from copy import deepcopy

    cube_out = deepcopy(cube_in)
    cube_out.data = ma.masked_where(mask.core_data() != 0, cube_in.core_data())
    return cube_out


def mask_cell(mask, cell, track, masked=False):
    """Create mask for specific cell.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tracked volumes 0 everywhere
        else).

    cell : int
        Integer id of cell to create masked cube for.

    track : pandas.DataFrame
        Output of the linking.

    masked : bool, optional
        Bool determining whether to mask the mask for the cell where
        it is 0. Default is False.

    Returns
    -------
    mask_i : numpy.ndarray
        Mask for a specific cell.
    """

    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i = mask_features(mask, feature_ids, masked=masked)
    return mask_i


def mask_cell_surface(mask, cell, track, masked=False, z_coord="model_level_number"):
    """Create surface projection of 3d-mask for individual cell by
    collapsing one coordinate.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes, 0 everywhere
        else).

    cell : int
        Integer id of cell to create masked cube for.

    track : pandas.DataFrame
        Output of the linking.

    masked : bool, optional
        Bool determining whether to mask the mask for the cell where
        it is 0. Default is False.

    z_coord : str, optional
        Name of the coordinate to collapse. Default is 'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube
        Collapsed Masked cube for the cell with the maximum value
        along the collapsed coordinate.

    """

    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i_surface = mask_features_surface(
        mask, feature_ids, masked=masked, z_coord=z_coord
    )
    return mask_i_surface


def mask_cube_features(variable_cube, mask, feature_ids):
    """Mask cube for tracked volume of one or more specific
    features.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes, 0 everywhere
        else).

    feature_ids : int or list of ints
        Integer ids of features to create masked cube for.

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube with data for respective features.
    """

    from dask.array import ma, isin
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        ~isin(mask.core_data(), feature_ids), variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_features(mask, feature_ids, masked=False):
    """Create mask for specific features.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    feature_ids : int or list of ints
        Integer ids of the features to create the masked cube for.

    masked : bool, optional
        Bool determining whether to mask the mask for the cell where
        it is 0. Default is False.

    Returns
    -------
    mask_i : numpy.ndarray
        Masked cube for specific features.
    """

    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_data = mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(), feature_ids)] = 0
    if masked:
        mask_i.data = ma.masked_equal(mask_i.core_data(), 0)
    return mask_i


def mask_features_surface(
    mask, feature_ids, masked=False, z_coord="model_level_number"
):
    """Create surface projection of 3d-mask for specific features
    by collapsing one coordinate.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    feature_ids : int or list of ints
        Integer ids of the features to create the masked cube for.

    masked : bool, optional
        Bool determining whether to mask the mask for the cell where
        it is 0. Default is False

    z_coord : str, optional
        Name of the coordinate to collapse. Default is
        'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube
        Collapsed Masked cube for the features with the maximum value
        along the collapsed coordinate.
    """

    from iris.analysis import MAX
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    #     mask_i.data=[~isin(mask_i.data,feature_ids)]=0
    mask_i_data = mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(), feature_ids)] = 0
    mask_i_surface = mask_i.collapsed(z_coord, MAX)
    if masked:
        mask_i_surface.data = ma.masked_equal(mask_i_surface.core_data(), 0)
    return mask_i_surface


def mask_all_surface(mask, masked=False, z_coord="model_level_number"):
    """Create surface projection of 3d-mask for all features
    by collapsing one coordinate.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    masked : bool, optional
        Bool determining whether to mask the mask for the cell where
        it is 0. Default is False

    z_coord : str, optional
        Name of the coordinate to collapse. Default is
        'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube (2D)
        Collapsed Masked cube for the features with the maximum value
        along the collapsed coordinate.
    """

    from iris.analysis import MAX
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_surface = mask_i.collapsed(z_coord, MAX)
    mask_i_surface_data = mask_i_surface.core_data()
    mask_i_surface.data[mask_i_surface_data > 0] = 1
    if masked:
        mask_i_surface.data = ma.masked_equal(mask_i_surface.core_data(), 0)
    return mask_i_surface


def convert_feature_mask_to_cells(
    features: pd.DataFrame, feature_mask: xr.DataArray, stubs: Optional[int] = None
) -> xr.DataArray:
    """Relabels a feature mask provided by tobac.segmentation with the cell
    values provided by tobac.linking_trackpy

    Parameters
    ----------
    features : pd.DataFrame
        A feature dataframe with cell values provided by tobac.linking_trackpy
    feature_mask : xr.DataArray
        A feature mask from tobac.segmentation corresponding to the features in
        the feature dataframe input
    stubs : int, optional (default: None)
        The stub values used for unlinked cells in tobac.linking_trackpy. If
        None, the stub cells with be relabelled with the stub cell value in the
        feature dataframe. If a value is provided, the masked regions
        corresponding to stub cells with be removed from the output. Warning:
        the presence of stub cells may make it impossible to perfectly
        reconstruct the feature mask afterwards as any stub features will be
        removed.

    Returns
    -------
    xr.DataArray
        A mask of cell regions corresponding to the cells in the input dataframe

    Raises
    ------
    ValueError
        If the features input does not have a cell column
    ValueError
        If there are labels in the feature_mask that are not present in the
        features dataframe
    """
    if "cell" not in features.columns:
        raise ValueError(
            "`cell` column not found in features input, please perform tracking on this data before converting features to cells"
        )

    cell_mask = feature_mask.copy()

    cell_mapper = xr.DataArray(
        features.cell.copy(), dims=("feature",), coords=dict(feature=features.feature)
    )

    if stubs is not None:
        cell_mapper.data[features.cell == stubs] = 0

    wh_nonzero_label = np.flatnonzero(cell_mask)

    try:
        cell_mask.data.ravel()[wh_nonzero_label] = cell_mapper.loc[
            feature_mask.values.ravel()[wh_nonzero_label]
        ]
    except KeyError:
        raise ValueError(
            "Values in feature_mask are not present in features, please ensure that you are using the correct feature_mask for the tracked features, and that any filtering has been applied to both the mask and features"
        )

    cell_mask = cell_mask.assign_attrs(dict(units="cell"))

    return cell_mask


def convert_cell_mask_to_features(
    features: pd.DataFrame,
    cell_mask: xr.DataArray,
    stubs: Optional[int] = None,
) -> xr.DataArray:
    """Relabels a cell mask, such as that produced by
    convert_feature_mask_to_cells, to the feature values provided by
    tobac.linking_trackpy

    Parameters
    ----------
    features : pd.DataFrame
        A feature dataframe with cell values provided by tobac.linking_trackpy
    cell_mask : xr.DataArray
        A cekk mask corresponding to the cells in the feature dataframe input
    stubs : int, optional (default: None)
        The stub values used for unlinked cells in tobac.linking_trackpy. If
        None, the stub cells with be relabelled with the stub cell value in the
        feature dataframe. If a value is provided, the masked regions
        corresponding to stub cells with be removed from the output. Warning:
        features with stub values will be set to zero in the output feature
        mask

    Returns
    -------
    xr.DataArray
        A mask of feature regions corresponding to the features in the input
        dataframe

    Raises
    ------
    ValueError
        If duplicate cell values are present at any timestep in the input
        dataframe and the stubs parameter is not provided
    ValueError
        If duplicate cell values are present at any timestep in the input
        dataframe that are not equal to the provided stubs value
    ValueError
        If cell_mask includes cell values not present in the input dataframe
    """
    feature_mask = cell_mask.copy()

    for i, _, mask_slice, features_slice in field_and_features_over_time(
        feature_mask, features
    ):

        if stubs is None:
            if np.any(features_slice.cell.duplicated(keep=False)):
                raise ValueError(
                    "Duplicate cell values found for a single timestep in features. This may be because there are stub cells included in the dataframe. If so, please provide these using the stubs parameter"
                )

            cell_mapper = xr.DataArray(
                features_slice.feature.copy(),
                dims=("feature",),
                coords=dict(feature=features_slice.cell.copy()),
            )

        else:
            features_slice = features_slice.copy()
            # Set feature label to 0 for stub cells
            features_slice.loc[features_slice.cell == stubs, ["feature"]] = 0

            if np.any(
                features_slice.loc[features_slice.feature != 0, ["cell"]].duplicated(
                    keep=False
                )
            ):
                raise ValueError(
                    "Duplicate cell values found for a single timestep in features that does not match the provided stub value. This may be because the stub value provided is incorrect."
                )

            features_slice = features_slice[~features_slice.duplicated()]

            cell_mapper = xr.DataArray(
                features_slice.feature.copy(),
                dims=("feature",),
                coords=dict(feature=features_slice.cell.copy()),
            )

        wh_nonzero_label = np.flatnonzero(mask_slice)

        try:
            feature_mask.data[i].ravel()[wh_nonzero_label] = cell_mapper.loc[
                mask_slice.values.ravel()[wh_nonzero_label]
            ]
        except KeyError:
            raise ValueError(
                "Cell values in cell_mask are not present in features, please ensure that you are using the correct cell_mask for the tracked features, and that any filtering has been applied to both the mask and features"
            )

    feature_mask = feature_mask.assign_attrs(dict(units="feature"))

    return feature_mask
