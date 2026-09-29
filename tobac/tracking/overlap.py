"""Provide overlap tracking methods"""

import datetime
from typing import Generator, Literal, Optional, Union

import cftime
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


def _get_paired_field_and_features_iterator(
    Mask: xr.DataArray, Features: pd.DataFrame
) -> Generator[
    tuple[
        tuple[
            int,
            Union[datetime.datetime, np.datetime64, cftime.datetime],
            xr.DataArray,
            pd.DataFrame,
        ],
        tuple[
            int,
            Union[datetime.datetime, np.datetime64, cftime.datetime],
            xr.DataArray,
            pd.DataFrame,
        ],
    ],
    None,
    None,
]:
    """Generator yielding mask and features for consecutive time step pairs.

    Returns the output of field_and_features_over_time for timesteps t and t+1.

    Parameters
    ----------
    Mask : xr.DataArray
        The mask to iterate over.
    Features : pd.DataFrame
        The features dataframe to iterate through.

    Yields
    ------
    tuple[tuple[int, Union[datetime.datetime, np.datetime64, cftime.datetime], xr.DataArray, pd.DataFrame], tuple[int, Union[datetime.datetime, np.datetime64, cftime.datetime], xr.DataArray, pd.DataFrame]]
        Each iteration yields two tuples containing:
        - Iteration index
        - Time value
        - Slice of field at that time
        - Slice of features with times within the time padding tolerance
    """

    origin_iterator = field_and_features_over_time(Mask, Features)
    destination_iterator = field_and_features_over_time(Mask, Features)
    _ = next(destination_iterator)
    return zip(origin_iterator, destination_iterator)


def _get_indices_from_labels(
    labels: np.ndarray,
) -> tuple[dict[int, np.ndarray[int]], dict[int, int]]:
    """Function to get the x, y, and z indices (as well as point count) of all labeled regions.
    Slightly less deranged than the version in internal utils.

    Parameters
    ----------
    labels : np.ndarray
        The array of labels to get the indices for.
    Returns
    -------
    counts : dict
        The number of points in the label number (key: label number).
    coordinates : dict
        The coordinates in the label number. This is either a 2 or 3 x n array for each label
    """

    counts = {}
    coordinates = {}

    # loop through all skimage identified regions
    region_props = skimage.measure.regionprops(labels)
    for region_prop in region_props:
        coordinates[region_prop.label] = region_prop.coords.T
        counts[region_prop.label] = region_prop.coords.shape[0]

    return counts, coordinates


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

    wh = np.logical_and(
        matched_counts >= min_count, matched_counts / counts >= relative_count
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
        # sort_args = np.argsort(i_ind)
        # i_ind = i_ind[sort_args]
        # j_ind = j_ind[sort_args]
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
    tracks: pd.DataFrame,
    label_coords: dict[int, np.ndarray[int]],
    delta_t: float,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, np.ndarray[int]]:
    """Translate label coordinates based on velocity and time step.

    Four translation methods are available:
    - None: no translation applied
    - constant: translate using a constant velocity specified by velocity_constant
    - drift: translate all features using the average velocity of cells tracked at the previous timestep
    - predict: translate cells individually according to their velocity at the previous time step. For newly initialised cells, one of the further 4 methods are used:
        - None: initialise with 0 velocity
        - constant: initialise with velocity_constant
        - mean: initialise with the average velocity of cells tracked at the previous timestep
        - nearest: initialise with the velocity of the nearest cell tracked at the previous timestep

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe containing velocity information.
    label_coords : dict[int, np.ndarray[int]]
        Dictionary of coordinate arrays for each label.
    delta_t : float
        Time step in seconds.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Translation method to apply. Default is None (no translation).
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Method for velocity estimation. Used with 'predict' method.
    velocity_constant : float or np.ndarray, optional
        Constant velocity value or array. Only used if translate_method=="constant" or velocity_method=="constant"
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    hdim1_size : int, optional
        Size of first horizontal dimension.
    hdim2_size : int, optional
        Size of second horizontal dimension.
    vdim_size : int, optional
        Size of vertical dimension.

    Returns
    -------
    dict[int, np.ndarray[int]]
        Dictionary of translated coordinate arrays for each label.

    """
    if translate_method == "predict":
        _update_predicted_velocities(
            tracks, velocity_method=velocity_method, velocity_constant=velocity_constant
        )
        return {
            k: _wrap_coords(
                (
                    label_coords[k].T
                    + np.round(tracks._track_velocity[k] * delta_t).astype(int)
                ).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    if translate_method == "drift":
        translation = np.round(tracks._track_velocity.mean() * delta_t).astype(int)
        return {
            k: _wrap_coords(
                (label_coords[k].T + translation).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    if translate_method == "constant":
        translation = np.round(velocity_constant * delta_t).astype(int)
        return {
            k: _wrap_coords(
                (label_coords[k].T + translation).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    return label_coords


def _find_overlaps(
    tracks: pd.DataFrame,
    origin_labels: xr.DataArray,
    destination_labels: xr.DataArray,
    min_count: int = 1,
    relative_count: float = 0,
    delta_t: int = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, int]:
    """Find optimal one-to-one label overlaps between two time steps.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe from the origin time step.
    origin_labels : xr.DataArray
        Label array at origin time step.
    destination_labels : xr.DataArray
        Label array at destination time step.
    min_count : int, optional
        Minimum overlap pixel count. Default is 1.
    relative_count : float, optional
        Minimum relative overlap fraction. Default is 0.
    delta_t : int, optional
        Time difference between steps in seconds. Default is 0.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Label translation method before overlap calculation.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Velocity estimation method for translation.
    velocity_constant : float or np.ndarray, optional
        Constant velocity for translation.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    hdim1_size : int, optional
        Size of first horizontal dimension.
    hdim2_size : int, optional
        Size of second horizontal dimension.
    vdim_size : int, optional
        Size of vertical dimension.

    Returns
    -------
    dict[int, int]
        One-to-one mapping of origin labels to destination labels.

    """
    label_counts, label_coords = _get_indices_from_labels(origin_labels.values)
    if translate_method is not None:
        label_coords = _translate_labels(
            tracks,
            label_coords,
            delta_t,
            translate_method=translate_method,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
            PBC_flag=PBC_flag,
            hdim1_size=hdim1_size,
            hdim2_size=hdim2_size,
            vdim_size=vdim_size,
        )
    overlap_candidates = {
        k: _find_overlaps_for_label(
            label_coords[k],
            label_counts[k],
            destination_labels,
            min_count=min_count,
            relative_count=relative_count,
        )
        for k in label_coords.keys()
    }
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

    """
    domain_size = np.array(domain_size) + 1
    pos_neg_offset = np.where(start_coords < end_coords, 1, -1)
    if len(domain_size) == 3:
        domain_size[0] = 0
    if PBC_flag in [None, "none", "hdim1"]:
        domain_size[-1] = 0
    if PBC_flag in [None, "none", "hdim2"]:
        domain_size[-2] = 0

    return pos_neg_offset * np.minimum(
        np.abs(end_coords - start_coords),
        np.abs(end_coords - pos_neg_offset * domain_size - start_coords),
    )


def _assign_velocities(
    tracks: pd.DataFrame,
    matches: dict[int, int],
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    prior: bool = False,
) -> None:
    """Calculate and assign velocities from matched feature positions.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with position and time columns.
    matches : dict[int, int]
        Mapping of origin to destination feature IDs.
    domain_size : tuple[int]
        Size of domain in each dimension.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    prior : bool, optional
        If True, assign velocity to origin features. If False, assign to
        destination features. Default is False.

    Returns
    -------
    None
        Modifies tracks DataFrame in place.

    """
    if "vdim" in tracks.columns:
        end_locations = tracks.loc[
            matches.values(), ["vdim", "hdim_1", "hdim_2"]
        ].to_numpy()
        start_locations = tracks.loc[
            matches.keys(), ["vdim", "hdim_1", "hdim_2"]
        ].to_numpy()
    else:
        end_locations = tracks.loc[matches.values(), ["hdim_1", "hdim_2"]].to_numpy()
        start_locations = tracks.loc[matches.keys(), ["hdim_1", "hdim_2"]].to_numpy()
    velocities = (
        _calc_distances_pbcs(
            start_locations, end_locations, domain_size, PBC_flag=PBC_flag
        )
        / (
            tracks.loc[matches.values(), ["time"]]
            - tracks.loc[matches.keys(), ["time"]].values
        )
        .time.dt.total_seconds()
        .to_numpy()[:, None]
    )
    if prior:
        tracks.loc[matches.keys(), "_track_velocity"] = dict(
            zip(matches.keys(), velocities)
        )
    else:
        tracks.loc[matches.values(), "_track_velocity"] = dict(
            zip(matches.values(), velocities)
        )


def _bootstrap_velocities(
    tracks: pd.DataFrame,
    mask: xr.DataArray,
    min_count: int = 1,
    relative_count: float = 0,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> None:
    """Initialize velocities from first pair of time steps.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe to populate with initial velocities.
    mask : xr.DataArray
        Label mask array over time.
    min_count : int, optional
        Minimum overlap pixel count. Default is 1.
    relative_count : float, optional
        Minimum relative overlap fraction. Default is 0.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    None
        Modifies tracks DataFrame in place.

    """
    [_, _, origin_labels, _], [_, _, destination_labels, _] = next(
        _get_paired_field_and_features_iterator(mask, tracks)
    )
    matched_overlaps = _find_overlaps(
        tracks,
        origin_labels,
        destination_labels,
        min_count=min_count,
        relative_count=relative_count,
        translate_method=None,
    )
    _assign_velocities(
        tracks, matched_overlaps, origin_labels.shape, PBC_flag=PBC_flag, prior=True
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

    hdim1_size = mask.shape[-2]
    hdim2_size = mask.shape[-1]
    vdim_size = mask.shape[-3] if "vdim" in features else None

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

    if translate_method in ["drift", "predict"]:
        _bootstrap_velocities(tracks, mask)

    paired_iterator = _get_paired_field_and_features_iterator(mask, tracks)

    for [_, origin_timestep, origin_labels, origin_features], [
        _,
        destination_timestep,
        destination_labels,
        _,
    ] in paired_iterator:
        delta_t = (
            to_timestamp(destination_timestep.values)
            - to_timestamp(origin_timestep.values)
        ).total_seconds()
        matches = _find_overlaps(
            origin_features,
            origin_labels,
            destination_labels,
            min_count=minimum_overlap,
            relative_count=minimum_relative_overlap,
            translate_method=translate_method,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
            delta_t=delta_t,
            hdim1_size=hdim1_size,
            hdim2_size=hdim2_size,
            vdim_size=vdim_size,
        )
        _assign_cells_to_matches(tracks, matches)
        if translate_method in ["drift", "predict"]:
            _assign_velocities(tracks, matches, origin_labels.shape, PBC_flag=PBC_flag)

    if "_track_velocity" in tracks.columns:
        tracks = tracks.drop("_track_velocity", axis=1)

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
