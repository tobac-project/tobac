"""Provide essential methods for masking

"""

import logging
from tobac.utils.convert import xarray_to_iris


@xarray_to_iris
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
        3D cube containing columns of 2D mask (int id for tacked
        volumes 0 everywhere else).
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


@xarray_to_iris
def mask_cube_cell(variable_cube, mask, cell, track):
    """Mask cube for tracked volume of an individual cell.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    cell : int
        Interger id of cell to create masked cube for.

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


@xarray_to_iris
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

    Notes
    -----
    unsure about short summary
    """

    from dask.array import ma
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        mask.core_data() == 0, variable_cube_out.core_data()
    )
    return variable_cube_out


@xarray_to_iris
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


@xarray_to_iris
def mask_cube(cube_in, mask):
    """Mask cube where mask (array) is larger than zero.

    Parameters
    ----------
    cube_in : iris.cube.Cube
        Unmasked data cube.

    mask : numpy.ndarray or dask.array
        Mask to use for masking, >0 where cube is supposed to be masked.

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube.
    """

    from dask.array import ma
    from copy import deepcopy

    cube_out = deepcopy(cube_in)
    cube_out.data = ma.masked_where(mask != 0, cube_in.core_data())
    return cube_out


@xarray_to_iris
def mask_cell(mask, cell, track, masked=False):
    """Create mask for specific cell.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    cell : int
        Interger id of cell to create masked cube for.

    track

    masked : bool, optional
        Default is False.

    Returns
    -------
    mask_i : numpy.ndarray
        Masked cube for untracked volume.

    Notes
    -----
    unsure about mask_i, track and masked
    """

    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i = mask_features(mask, feature_ids, masked=masked)
    return mask_i


@xarray_to_iris
def mask_cell_surface(mask, cell, track, masked=False, z_coord="model_level_number"):
    """Create surface projection of mask for individual cell.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    cell : int
        Interger id of cell to create masked cube for.

    track

    masked : bool, optional
        Default is False.

    z_coord : str, optional
        Default is 'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube
        Masked cube for untracked volume.

    Notes
    -----
    unsure about Returns
    """

    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i_surface = mask_features_surface(
        mask, feature_ids, masked=masked, z_coord=z_coord
    )
    return mask_i_surface


@xarray_to_iris
def mask_cell_columns(mask, cell, track, masked=False, z_coord="model_level_number"):
    """Create mask with entire columns for individual cell.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    cell : int
        Interger id of cell to create masked cube for.

    track

    masked : bool, optional
        Default is False.

    z_coord : str, optional
        Default is 'model_level_number'.

    Returns
    -------
    mask_i : iris.cube.Cube
        Masked cube for untracked volume.
    """

    feature_ids = track.loc[track["cell"] == cell].loc["feature"]
    mask_i = mask_features_columns(mask, feature_ids, masked=masked, z_coord=z_coord)
    return mask_i


@xarray_to_iris
def mask_cube_features(variable_cube, mask, feature_ids):
    """Mask cube for tracked volume of an individual cell.

    Parameters
    ----------
    variable_cube : iris.cube.Cube
        Unmasked data cube.

    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    cell : int
        Interger id of cell to create masked cube for.

    feature_ids : int

    Returns
    -------
    variable_cube_out : iris.cube.Cube
        Masked cube with data for respective cell.
    """

    from dask.array import ma, isin
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        ~isin(mask.core_data(), feature_ids), variable_cube_out.core_data()
    )
    return variable_cube_out


@xarray_to_iris
def mask_features(mask, feature_ids, masked=False):
    """Create mask for specific features.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    feature_ids : int

    masked : bool, optional
        Default is False.

    Returns
    -------
    mask_i : numpy.ndarray
        Masked cube for untracked volume.

    Notes
    -----
    unsure about Returns, Parameters need more descriptions
    """

    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_data = mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(), feature_ids)] = 0
    if masked:
        mask_i.data = ma.masked_equal(mask_i.core_data(), 0)
    return mask_i


@xarray_to_iris
def mask_features_surface(
    mask, feature_ids, masked=False, z_coord="model_level_number"
):
    """Create surface mask for individual features.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    feature_ids : int

    masked : bool, optional
        Default is False.

    z_coord : str, optional
        Name of the vertical coordinate in the cube. Default is
        'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube
        Masked cube for untracked volume.

    Notes
    -----
    unsure about Returns
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


@xarray_to_iris
def mask_all_surface(mask, masked=False, z_coord="model_level_number"):
    """Create surface mask for individual features.

    Parameters
    ----------
    mask : iris.cube.Cube
        Cube containing mask (int id for tacked volumes 0 everywhere
        else).

    masked : bool, optional
        Default is False.

    z_coord : str, optional
        Name of the vertical coordinate in the cube. Default is
        'model_level_number'.

    Returns
    -------
    mask_i_surface : iris.cube.Cube (2D)
        Mask with 1 below features and 0 everywhere else.
    """

    from iris.analysis import MAX
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_surface = mask_i.collapsed(z_coord, MAX)
    mask_i_surface_data = mask_i_surface.core_data()
    mask_i_surface[mask_i_surface_data > 0] = 1
    if masked:
        mask_i_surface.data = ma.masked_equal(mask_i_surface.core_data(), 0)
    return mask_i_surface


# @xarray_to_iris
# def mask_features_columns(mask,feature_ids,masked=False,z_coord='model_level_number'):
#     ''' Mask cube for untracked volume
#     Input:
#     variable_cube:     iris.cube.Cube
#                        unmasked data cube
#     mask:              iris.cube.Cube
#                        cube containing mask (int id for tacked volumes 0 everywhere else)
#     Output:
#     variable_cube_out: iris.cube.Cube
#                        Masked cube for untracked volume
#     '''
#     from iris.analysis import MAX
#     import numpy as np
#     from copy import deepcopy
#     mask_i=deepcopy(mask)
#     mask_i.data[~np.isin(mask_i.data,feature_ids)]=0
#     mask_i_surface=mask_i.collapsed(z_coord,MAX)
#     for cube_slice in mask_i.slices(['time','x','y']):
#             cube_slice.data=mask_i_surface.core_data()
#     if masked:
#         mask_i.data=np.ma.array(mask_i.data,mask=mask_i.data)
#     return mask_i
