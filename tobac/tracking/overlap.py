"""Provide overlap tracking methods"""

from functools import partial
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import skimage.measure
from scipy.sparse import coo_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.neighbors import BallTree

import tobac.utils.internal as internal_utils
from tobac.utils.datetime import to_timestamp
from tobac.utils.generators import field_and_features_over_time
from tobac.utils.periodic_boundaries import build_distance_function

from tobac.feature_detection import feature_position


def _unique_nonzero(arr: np.ndarray, **kwargs) -> np.ndarray:
    """Return unique nonzero elements of an array.

    Equivalent to np.unique on nonzero elements. Output is always flattened,
    and inverse indices are positions within the nonzero array, not the original.

    Parameters
    ----------
    arr : np.ndarray
        Array to find unique nonzero values in.
    **kwargs
        Additional keyword arguments passed to np.unique.

    Returns
    -------
    np.ndarray
        Unique nonzero values in the input array.

    """
    return np.unique(arr[arr != 0], **kwargs)


def _find_overlaps_for_label(
    coords: np.ndarray[int],
    counts: int,
    destination_labels: np.ndarray[int],
    destination_counts: dict[int, int],
    min_count: int = 1,
    relative_count: float = 0,
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    """Find unique, nonzero label overlaps for a given set of array indices.

    Parameters
    ----------
    coords : np.ndarray[int]
        Coordinate array of region locations. Must be 2xN or 3xN shaped
        for 2D or 3D data respectively.
    counts : int
        Number of pixels in the search region.
    destination_labels : np.ndarray[int]
        Array of label values to search for overlaps within.
    destination_counts : dict[int, int]
        Number of pixels in each label within destination_labels.
    min_count : int, optional
        Minimum number of pixels for a label to be returned as an overlap.
        Default is 1.
    relative_count : float, optional
        Minimum fraction of search region for a label to cover to be returned
        as an overlap. Default is 0.

    Returns
    -------
    matched_labels : np.ndarray[int]
        Array of labels matching the overlap criteria.
    matched_counts : np.ndarray[int]
        Array of pixel counts for each matched label.

    """
    matched_labels, matched_counts = _unique_nonzero(
        destination_labels.values[*coords], return_counts=True
    )

    min_counts = np.minimum(counts, [destination_counts[k] for k in matched_labels])

    wh = np.logical_and(
        matched_counts >= min_count, matched_counts / min_counts >= relative_count
    )

    return matched_labels[wh], matched_counts[wh]


def _maximise_matching_overlaps(
    overlaps: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, int]:
    """Find the optimum one-to-one feature matching using max weight matching.

    Uses the scipy min_weight_full_bipartite_matching maximum weight matching algorithm to optimise the one-to-one mapping
    of features to maximise total overlap area between timesteps.

    Parameters
    ----------
    overlaps : dict[int, tuple[np.ndarray, np.ndarray]]
        Dictionary of overlap candidates for each feature. Keys are feature IDs,
        values are tuples of (labels, counts) arrays.

    Returns
    -------
    dict[int, int]
        One-to-one mapping of origin features to destination features.

    """
    filtered_overlaps = {k: v for k, v in overlaps.items() if len(v[0]) > 0}
    # if no overlap candidates, return an empty dict
    if not len(filtered_overlaps):
        return {}
    origin_nodes = np.repeat(
        list(filtered_overlaps.keys()), [len(v[0]) for v in filtered_overlaps.values()]
    )
    i_map, i_nodes = np.unique(origin_nodes, return_inverse=True)
    destination_nodes, weights = np.concatenate(
        list(filtered_overlaps.values()), axis=1
    )
    j_map, j_nodes = np.unique(destination_nodes, return_inverse=True)

    total_nodes = j_nodes.max() + 1

    # need to add null "padding" nodes to ensure maximum matching is possible
    padding_nodes = np.arange(total_nodes, total_nodes + i_nodes.max() + 1, dtype=int)
    size = total_nodes + padding_nodes.size

    weights = np.concatenate([weights, np.full((padding_nodes.size,), -1e-15)])
    i = np.concatenate([i_nodes, np.unique(i_nodes)])
    j = np.concatenate([j_nodes, padding_nodes])

    sparse_graph = coo_array(
        (weights, (i, j)),
        shape=(i_nodes.max() + 1, size),
    )

    # maximal_matching = maximum_bipartite_matching(sparse_graph.T)

    i_ind, j_ind = min_weight_full_bipartite_matching(sparse_graph, maximize=True)

    if i_ind.size:
        # remove any null nodes
        wh_null = j_ind >= total_nodes
        i_ind = i_ind[~wh_null]
        j_ind = j_ind[~wh_null]
        return dict(zip(i_map[i_ind], j_map[j_ind]))

    # if no matches return empty dict
    return {}


class FeatureBallTree(BallTree):
    """A child class of scikit-learn's BallTree that handles feature dataframe input
    and periodic boundary conditions (PBC).
    """

    def __init__(
        self,
        features: pd.DataFrame,
        PBC_flag: Union[None, Literal["none", "hdim_1", "hdim_2", "both"]] = None,
        min_h1: int = 0,
        max_h1: int = 0,
        min_h2: int = 0,
        max_h2: int = 0,
        **kwargs,
    ) -> None:
        """Initialize a FeatureBallTree from a features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with columns 'hdim_1', 'hdim_2', and optionally 'vdim'
            for 2D and 3D data respectively.
        PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
            Specification of which dimensions have periodic boundary conditions.
            Default is None (no periodic boundaries).
        min_h1 : int, optional
            Minimum value of first horizontal dimension for PBC. Default is 0.
        max_h1 : int, optional
            Maximum value of first horizontal dimension for PBC. Default is 0.
        min_h2 : int, optional
            Minimum value of second horizontal dimension for PBC. Default is 0.
        max_h2 : int, optional
            Maximum value of second horizontal dimension for PBC. Default is 0.
        **kwargs
            Additional keyword arguments passed to BallTree constructor.

        """
        self.is_3D = "vdim" in features.columns
        self.index = features.index.values
        if PBC_flag in ["hdim_1", "hdim_2", "both"]:
            kwargs["metric"] = "pyfunc"
            kwargs["func"] = build_distance_function(
                min_h1, max_h1, min_h2, max_h2, PBC_flag, self.is_3D
            )
        super().__init__(self._get_feature_locations(features), **kwargs)

    def _get_feature_locations(self, features: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with position columns.

        Returns
        -------
        np.ndarray
            Coordinate array of shape (n_features, n_dims) containing either
            ['hdim_1', 'hdim_2'] for 2D or ['vdim', 'hdim_1', 'hdim_2'] for 3D data.

        Raises
        ------
        AssertionError
            If dimensionality of query features does not match the tree dimensionality.

        """
        assert (
            "vdim" in features.columns
        ) == self.is_3D, "Query features must match dimensionality of original features"
        return (
            features[["vdim", "hdim_1", "hdim_2"]].to_numpy()
            if self.is_3D
            else features[["hdim_1", "hdim_2"]].to_numpy()
        )

    def query(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Query the tree for nearest neighbors.

        Wraps the parent BallTree.query method to handle feature dataframe input
        and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query.
        **kwargs
            Keyword arguments passed to BallTree.query.

        Returns
        -------
        indices : np.ndarray
            Feature indices of nearest neighbors, shape (n_queries,) or
            (n_queries, k) depending on input.
        distances : np.ndarray, optional
            Distances to nearest neighbors if return_distance=True in kwargs.

        """
        query_result = super().query(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                self.index[query_result[1]],
                query_result[0],
            )  # flip around results to match query_radius
        return self.index[query_result]

    def query_radius(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Query the tree for neighbors within a specified radius.

        Wraps the parent BallTree.query_radius method to handle feature dataframe
        input and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query_radius.
        **kwargs
            Keyword arguments passed to BallTree.query_radius.

        Returns
        -------
        neighbors : np.ndarray
            Array of feature indices of neighbors within radius for each query point.
            Shape is (n_queries,) with dtype=object containing variable-length arrays.
        distances : np.ndarray, optional
            Distances to neighbors if return_distance=True in kwargs.

        """
        query_result = super().query_radius(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                np.array([self.index[inds] for inds in query_result[0]], dtype=object),
                query_result[1],
            )
        return np.array([self.index[inds] for inds in query_result], dtype=object)


def _update_predicted_velocities(
    features: pd.DataFrame,
    velocity_method: Union[None, Literal["constant", "mean", "nearest"]] = "constant",
    velocity_constant: Union[None, float, np.ndarray] = 0,
) -> None:
    """Update predicted velocities for features with missing velocity values.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe with '_track_velocity' column to update.
    velocity_method : {None, 'constant', 'mean', 'nearest'}, optional
        Method for filling missing velocities. Default is 'constant'.
        - 'constant': Use velocity_constant value
        - 'mean': Use mean of existing velocities
        - 'nearest': Use nearest neighbor velocity (accounting for PBCs)
    velocity_constant : None, float, or np.ndarray, optional
        Constant velocity value or array of values. Default is 0.

    Returns
    -------
    None
        Modifies features DataFrame in place.

    """
    wh_missing_vels = features._track_velocity.isna()
    if wh_missing_vels.any():
        if velocity_method == "constant":
            features.loc[wh_missing_vels, "_track_velocity"] = (
                {k: velocity_constant for k in features[wh_missing_vels].index}
                if hasattr(velocity_constant, "__iter__")
                else velocity_constant
            )
        elif velocity_method is None or wh_missing_vels.all():
            features.loc[wh_missing_vels, "_track_velocity"] = 0
        elif velocity_method == "mean":
            mean_vel = features._track_velocity.mean()
            features.loc[wh_missing_vels, "_track_velocity"] = {
                k: mean_vel for k in features[wh_missing_vels].index
            }
        elif velocity_method == "nearest":
            # create BallTree to find nearest velocity accounting for PBCs
            btree = FeatureBallTree(features[~wh_missing_vels])
            features.loc[wh_missing_vels, "_track_velocity"] = features._track_velocity[
                ~wh_missing_vels
            ][
                btree.query(features[wh_missing_vels], return_distance=False).ravel()
            ].values


def _wrap_coords(
    coords: np.ndarray[int],
    hdim1_size: int,
    hdim2_size: int,
    vdim_size: Optional[int] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[int]:
    """Wrap coordinates according to periodic boundary conditions. Coordinate locations which fall outside the size of the array are removed.

    Parameters
    ----------
    coords : np.ndarray[int]
        Coordinate array to wrap. Shape is (n_dims, n_points).
    hdim1_size : int
        Size of the first horizontal dimension.
    hdim2_size : int
        Size of the second horizontal dimension.
    vdim_size : int, optional
        Size of the vertical dimension. If None, no vertical wrapping applied.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Specification of which dimensions have periodic boundaries.
        Default is None.

    Returns
    -------
    np.ndarray[int]
        Wrapped coordinate array with invalid points filtered out.

    """
    if PBC_flag in ["hdim_1", "both"]:
        coords[-2] = coords[-2] % hdim1_size
    if PBC_flag in ["hdim_2", "both"]:
        coords[-1] = coords[-1] % hdim2_size
    filter = np.logical_and(coords[-2] < hdim1_size, coords[-1] < hdim2_size)
    if vdim_size is not None:
        filter = np.logical_and(filter, coords[0] < vdim_size)
    filter = np.logical_and(filter, np.any(coords >= 0, axis=0))
    return coords[:, filter]


def _translate_labels(
    features: pd.DataFrame,
    timestep: pd.Timestamp,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, np.ndarray[int]]:
    """Translate feature coordinates to a future timestep.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe containing position and velocity information.
    timestep : pd.Timestamp
        Timestamp for the target timestep.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Translation strategy to apply to coordinates. Defaults to None, which
        leaves the coordinates unchanged.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Velocity estimation method used when ``translate_method='predict'``.
    velocity_constant : float or np.ndarray, optional
        Constant velocity used when ``translate_method='constant'``.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    hdim1_size : int, optional
        Size of the first horizontal dimension.
    hdim2_size : int, optional
        Size of the second horizontal dimension.
    vdim_size : int, optional
        Size of the vertical dimension for 3D data.

    Returns
    -------
    dict[int, np.ndarray[int]]
        Mapping of feature indices to translated coordinate arrays. The arrays are
        stored in the ``_translated_coords`` column of ``features`` in-place.

    """

    wrap_coords = partial(
        _wrap_coords,
        hdim1_size=hdim1_size,
        hdim2_size=hdim2_size,
        vdim_size=vdim_size,
        PBC_flag=PBC_flag,
    )
    if translate_method == "predict":
        _update_predicted_velocities(
            features,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
        )
        features["_translated_coords"] = (
            features._coords
            + features._track_velocity
            * (
                to_timestamp(timestep.values) - to_timestamp(features.time)
            ).dt.total_seconds()
        ).apply(lambda a: wrap_coords(a.round().astype(int).T))
    elif translate_method == "drift":
        if (~features._track_velocity.isna()).any():
            features["_translated_coords"] = (
                features._coords
                + pd.Series(
                    dict(
                        zip(
                            features.index,
                            features._track_velocity.mean()
                            * (
                                to_timestamp(timestep.values)
                                - to_timestamp(features.time)
                            )
                            .dt.total_seconds()
                            .to_numpy()[:, np.newaxis],
                        )
                    )
                )
            ).apply(lambda a: wrap_coords(a.round().astype(int).T))
        else:  # If no valid velocities
            features["_translated_coords"] = features._coords.apply(
                lambda a: a.round().astype(int).T
            )
    elif translate_method == "constant":
        features["_translated_coords"] = (
            features._coords
            + pd.Series(
                dict(
                    zip(
                        features.index,
                        velocity_constant
                        * (to_timestamp(timestep.values) - to_timestamp(features.time))
                        .dt.total_seconds()
                        .to_numpy()[:, np.newaxis],
                    )
                )
            )
        ).apply(lambda a: wrap_coords(a.round().astype(int).T))
    else:
        features["_translated_coords"] = features._coords.apply(
            lambda a: a.round().astype(int).T
        )


def _find_overlaps(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    destination_labels: xr.DataArray,
    timestep: pd.Timestamp,
    min_count: int = 1,
    relative_count: float = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> dict[int, int]:
    """Find overlap matches between origin and destination features.

    Parameters
    ----------
    origin_features : pd.DataFrame
        Features in the current timestep to be linked forward.
    destination_features : pd.DataFrame
        Features in the destination timestep.
    destination_labels : xr.DataArray
        Label mask for the destination timestep.
    timestep : pd.Timestamp
        Timestamp associated with the destination timestep.
    min_count : int, optional
        Minimum pixel overlap required for a match. Default is 1.
    relative_count : float, optional
        Minimum relative overlap fraction required for a match. Default is 0.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Translation method used to predict the next location of origin features.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Velocity estimation method for ``translate_method='predict'``.
    velocity_constant : float or np.ndarray, optional
        Constant velocity used when ``translate_method='constant'``.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    dict[int, int]
        Mapping of origin feature IDs to destination feature IDs for valid
        overlapping matches.

    """

    _translate_labels(
        origin_features,
        timestep,
        translate_method=translate_method,
        velocity_method=velocity_method,
        velocity_constant=velocity_constant,
        PBC_flag=PBC_flag,
        hdim1_size=destination_labels.shape[-2],
        hdim2_size=destination_labels.shape[-1],
        vdim_size=(
            destination_labels.shape[0] if len(destination_labels.shape) == 3 else None
        ),
    )
    overlap_candidates = origin_features.apply(
        lambda row: _find_overlaps_for_label(
            row._translated_coords,
            row._count,
            destination_labels,
            destination_features._count,
            min_count=min_count,
            relative_count=relative_count,
        ),
        axis=1,
    )
    return _maximise_matching_overlaps(overlap_candidates)


def _assign_cells_to_matches(tracks: pd.DataFrame, matches: dict[int, int]) -> None:
    """Assign cell identities to matched features.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' column to update.
    matches : dict[int, int]
        Mapping of origin feature IDs to destination feature IDs.

    Returns
    -------
    None
        Modifies tracks DataFrame in place.

    """
    prior_cells = tracks.loc[matches.keys(), "cell"]
    wh_unassigned = prior_cells == 0
    prior_cells[wh_unassigned] = (
        np.arange(wh_unassigned.sum(), dtype=int) + tracks.cell.max() + 1
    )
    tracks.loc[matches.keys(), "cell"] = prior_cells.values
    tracks.loc[matches.values(), "cell"] = prior_cells.values


def _calc_distances_pbcs(
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[float]:
    """Calculate distances between coordinate pairs accounting for PBCs.

    Parameters
    ----------
    start_coords : np.ndarray
        Starting coordinates. Shape is (n_points, n_dims).
    end_coords : np.ndarray
        Ending coordinates. Shape is (n_points, n_dims).
    domain_size : tuple[int]
        Size of domain in each dimension.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    np.ndarray[float]
        Distance vectors accounting for periodic boundaries.
        Shape is (n_points, n_dims).

    #"""
    if PBC_flag in [None, "none"]:
        return end_coords - start_coords
    domain_size = np.array(domain_size) + 1
    pos_neg_offset = np.where(start_coords < end_coords, 1, -1)
    if len(domain_size) == 3:
        domain_size[0] = 0
    if PBC_flag == "hdim1":
        domain_size[-1] = 0
    if PBC_flag == "hdim2":
        domain_size[-2] = 0

    non_pbc_dist = np.abs(end_coords - start_coords)
    pbc_dist = np.abs(end_coords - pos_neg_offset * domain_size - start_coords)

    return pos_neg_offset * np.where(non_pbc_dist <= pbc_dist, non_pbc_dist, -pbc_dist)


def _assign_velocities(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    matches: dict[int, int],
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    prior: bool = False,
) -> None:
    if len(matches):
        velocities = (
            _calc_distances_pbcs(
                np.stack(origin_features.loc[matches.keys(), "_centroid"]),
                np.stack(destination_features.loc[matches.values(), "_centroid"]),
                domain_size=domain_size,
                PBC_flag=PBC_flag,
            )
            / np.stack(
                (
                    destination_features.loc[matches.values(), "time"]
                    - origin_features.loc[matches.keys(), "time"].values
                ).dt.total_seconds()
            )[:, np.newaxis]
        )
        if prior:
            origin_features.loc[matches.keys(), "_track_velocity"] = dict(
                zip(matches.keys(), velocities)
            )
        else:
            destination_features.loc[matches.values(), "_track_velocity"] = dict(
                zip(matches.values(), velocities)
            )
    else:
        if prior:
            origin_features["_track_velocity"] = None
        else:
            destination_features["_track_velocity"] = None


def _bootstrap_velocities(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    destination_labels: xr.DataArray,
    timestep: pd.Timestamp,
    min_count: int = 1,
    relative_count: float = 0,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> None:
    """Estimate initial feature velocities from overlap matches.

    Parameters
    ----------
    origin_features : pd.DataFrame
        Features in the current timestep.
    destination_features : pd.DataFrame
        Features in the destination timestep.
    destination_labels : xr.DataArray
        Label mask for the destination timestep.
    timestep : pd.Timestamp
        Timestamp associated with the destination timestep.
    min_count : int, optional
        Minimum pixel overlap required for a match. Default is 1.
    relative_count : float, optional
        Minimum relative overlap fraction required for a match. Default is 0.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    None
        Updates the ``_track_velocity`` column of the features dataframes in-place.

    """
    matched_overlaps = _find_overlaps(
        origin_features,
        destination_features,
        destination_labels,
        timestep,
        min_count=min_count,
        relative_count=relative_count,
    )
    _assign_velocities(
        origin_features,
        destination_features,
        matched_overlaps,
        destination_labels.shape,
        PBC_flag=PBC_flag,
        prior=True,
    )


def _filter_stub_cells(
    tracks: pd.DataFrame,
    stubs: int,
    cell_number_start: int,
    cell_number_unassigned: int,
) -> pd.DataFrame:
    """Remove cells with fewer than minimum number of time steps.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' column.
    stubs : int
        Minimum number of time steps for a cell to be retained.
    cell_number_start : int
        Starting value for cell ID numbering.
    cell_number_unassigned : int
        Value to assign to unassigned/filtered cells.

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with stub cells removed and cell IDs renumbered.

    """
    # Ensure all cell values are zero or greater (they should be already)
    tracks["cell"] = np.maximum(tracks.cell.values, 0)
    if stubs > 1:
        cell_count = tracks.groupby("cell").cell.count()
        stub_cells = cell_count.index[cell_count < stubs].values
        tracks.loc[np.isin(tracks.cell, stub_cells), "cell"] = 0

    new_cells = np.unique(tracks.cell, return_inverse=True)[1]
    if (tracks.cell == 0).any():
        new_cells = np.where(
            new_cells > 0, new_cells + cell_number_start - 1, cell_number_unassigned
        )
    else:
        new_cells = new_cells + cell_number_start

    tracks["cell"] = new_cells

    return tracks


def _assign_cell_times(
    tracks: pd.DataFrame, cell_number_unassigned: int
) -> pd.DataFrame:
    """Calculate time since start of each cell.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' and 'time' columns.
    cell_number_unassigned : int
        Value assigned to unassigned cells.

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with new 'time_cell' column containing time
        relative to cell start.

    """
    tracks["time_cell"] = (
        tracks.time - tracks.groupby("cell").time.min()[tracks.cell.values].values
    )
    tracks.loc[tracks.cell == cell_number_unassigned, "time_cell"] = pd.Timedelta("nat")
    return tracks


def _get_coords_and_centroids(
    features: pd.DataFrame,
    mask: xr.DataArray,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> pd.DataFrame:
    """Attach region counts and centroids to feature entries.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe containing feature metadata.
    mask : xr.DataArray
        Labeled mask for the current timestep.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    pd.DataFrame
        Input features dataframe joined with ``_count``, ``_coords``, and
        ``_centroid`` columns derived from the mask.

    """
    if len(features):
        if len(mask.shape) == 3:  # is_3D
            feature_position_partial = partial(
                feature_position,
                hdim1_max=mask.shape[1],
                hdim2_max=mask.shape[2],
                PBC_flag=PBC_flag,
            )
            fp_lambda = lambda row: np.array(
                feature_position_partial(
                    row._coords[:, 1], row._coords[:, 2], vdim_indices=row._coords[:, 0]
                )
            )
        else:
            feature_position_partial = partial(
                feature_position,
                hdim1_max=mask.shape[0],
                hdim2_max=mask.shape[1],
                PBC_flag=PBC_flag,
            )
            fp_lambda = lambda row: np.array(feature_position_partial(*row._coords.T))

        props_df = (
            pd.DataFrame(
                skimage.measure.regionprops_table(
                    mask.values, properties=("label", "area", "coords")
                )
            )
            .rename(columns=dict(label="feature", area="_count", coords="_coords"))
            .set_index("feature")
        )

        props_df["_centroid"] = props_df.apply(fp_lambda, axis=1)
    else:
        props_df = pd.DataFrame(columns=["_count", "_coords", "_centroid"])

    return features.join(props_df)


def linking_overlap(
    features: pd.DataFrame,
    mask: xr.DataArray,
    stubs: int = 1,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    minimum_overlap: int = 1,
    minimum_relative_overlap: float = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    vertical_axis: Optional[int] = None,
    vertical_coord: Optional[str] = None,
    memory: int = 0,
) -> pd.DataFrame:
    """Link features through time using spatial overlap tracking.

    Tracks features across consecutive time steps by finding the best
    one-to-one matches based on spatial overlap, optionally translating
    features based on velocity predictions.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe with 'feature' index and position columns
        ('hdim_1', 'hdim_2', and optionally 'vdim').
    mask : xr.DataArray
        Time series of labeled feature masks with coordinates matching features.
    stubs : int, optional
        Minimum number of timesteps for a cell to be retained. Default is 1.
    cell_number_start : int, optional
        Starting value for cell numbering. Default is 1.
    cell_number_unassigned : int, optional
        Value for unassigned features. Default is -1.
    minimum_overlap : int, optional
        Minimum pixel overlap required for feature linking. Default is 1.
    minimum_relative_overlap : float, optional
        Minimum fraction of feature area required for linking. Default is 0.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Method for predicting feature positions at next time step. Default is None.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Method for velocity estimation if translate_method is used.
    velocity_constant : float or np.ndarray, optional
        Constant velocity for translation. Used if translate_method is 'constant'.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    vertical_coord : str, optional
        Name of the vertical coordinate. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string.
    vertical_axis : int, optional
        The vertical axis number of the data. If None, uses vertical_coord
        to determine axis. This must be >=0.
    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        .. warning :: This parameter should be used with caution, as it
                        can lead to erroneous trajectory linking,
                        especially for data with low time resolution.

    Returns
    -------
    pd.DataFrame
        Features dataframe with added columns:
        - 'cell': Cell identifier for tracked features
        - 'time_cell': Time relative to start of cell
        Features not linked to any cell are assigned cell_number_unassigned.

    """
    tracks = features.copy().set_index("feature")
    tracks["cell"] = 0

    time_axis = internal_utils.find_axis_from_coord(mask, "time")
    if len(mask.shape) == 4:
        if vertical_axis is None:
            # We need to determine vertical axis.
            # first, find the name of the vertical axis
            vertical_axis_name = internal_utils.find_vertical_coord_name(
                mask, vertical_coord=vertical_coord
            )
            # then find our axis number.
            vertical_axis = internal_utils.find_axis_from_coord(
                mask, vertical_axis_name
            )

            if vertical_axis is None:
                raise ValueError("Cannot find vertical coordinate.")

        if vertical_axis < 0:
            raise ValueError("vertical_axis must be >=0.")

        hdim_1_axis, hdim_2_axis = internal_utils.find_hdim_axes_3D(
            mask, vertical_axis=vertical_axis
        )

        mask = mask.transpose(
            *[
                mask.dims[i]
                for i in [time_axis, vertical_axis, hdim_1_axis, hdim_2_axis]
            ]
        )

    field_and_features = iter(field_and_features_over_time(mask, tracks))

    _, _, labels, features_t = next(field_and_features)
    origin_features = _get_coords_and_centroids(features_t, labels, PBC_flag=PBC_flag)

    try:
        frame, timestep, labels, features_t = next(field_and_features)
    except StopIteration:  # Only one timestep
        tracks = _assign_cell_times(
            tracks, cell_number_unassigned=cell_number_unassigned
        )
        # Reset index to match features input and replace features column in the correct location
        tracks = tracks.set_index(features.index)
        tracks.insert(features.columns.get_loc("feature"), "feature", features.feature)
        return tracks

    destination_features = _get_coords_and_centroids(
        features_t, labels, PBC_flag=PBC_flag
    )

    bootstrap = True if translate_method in ["drift", "predict"] else False

    while True:
        if len(origin_features) and len(destination_features):
            if bootstrap:
                _bootstrap_velocities(
                    origin_features,
                    destination_features,
                    labels,
                    timestep,
                    min_count=minimum_overlap,
                    relative_count=minimum_relative_overlap,
                    PBC_flag=PBC_flag,
                )
                bootstrap = False

            matched_overlaps = _find_overlaps(
                origin_features,
                destination_features,
                labels,
                timestep,
                min_count=minimum_overlap,
                relative_count=minimum_relative_overlap,
                translate_method=translate_method,
                velocity_method=velocity_method,
                velocity_constant=velocity_constant,
                PBC_flag=PBC_flag,
            )
            _assign_cells_to_matches(tracks, matched_overlaps)
            if translate_method in ["drift", "predict"]:
                _assign_velocities(
                    origin_features,
                    destination_features,
                    matched_overlaps,
                    labels.shape,
                    PBC_flag=PBC_flag,
                )
        else:
            if not len(origin_features):
                bootstrap = True  # if no features, we need to bootstrap next time we have features

        # construct new origin_features

        if (memory > 0) and len(origin_features):
            origin_features = pd.concat(
                [
                    origin_features[
                        np.logical_and(
                            (frame - origin_features.frame) <= memory,
                            np.logical_not(
                                np.isin(
                                    origin_features.index.values,
                                    list(matched_overlaps.keys()),
                                )
                            ),
                        )
                    ],
                    destination_features,
                ],
            )
        else:
            origin_features = destination_features

        try:
            frame, timestep, labels, features_t = next(field_and_features)
        except StopIteration:
            break
        destination_features = _get_coords_and_centroids(
            features_t, labels, PBC_flag=PBC_flag
        )

    tracks = _filter_stub_cells(
        tracks,
        stubs=stubs,
        cell_number_start=cell_number_start,
        cell_number_unassigned=cell_number_unassigned,
    )

    tracks = _assign_cell_times(tracks, cell_number_unassigned=cell_number_unassigned)

    # Reset index to match features input and replace features column in the correct location
    tracks = tracks.set_index(features.index)
    tracks.insert(features.columns.get_loc("feature"), "feature", features.feature)

    return tracks
