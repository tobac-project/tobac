# Overlap Tracking

## Overview

The default tracking method in *tobac*, {py:func}`tobac.tracking.linking_trackpy`, links features based only on their point-like positions (e.g., the weighted centroid), searching for candidate matches within a search radius around the (predicted) position of each feature (see {doc}`/getting_started/tracking_basics`).

{py:func}`tobac.tracking.linking_overlap` is an alternative tracking method that instead uses the **full spatial extent of each feature**. In addition to the features DataFrame, it requires a labeled `mask` of the features over time, in which each grid cell (pixel in 2D, voxel in 3D) belonging to a feature is labeled with that feature's `feature` value. Such a mask can be obtained directly from feature detection by passing `return_labels=True` to {py:func}`tobac.feature_detection.feature_detection_multithreshold`, or from segmentation (e.g., {py:func}`tobac.segmentation.segmentation`), whose output mask labels each segmented region with its feature ID.

For each pair of consecutive timesteps, the overlap between the mask of every feature in the current timestep and the masks of all features in the next timestep is computed. If a feature overlaps with several features in the next timestep (or vice versa), the ambiguity is resolved by choosing the one-to-one assignment between features that **maximizes the total overlap area**, using a maximum-weight bipartite matching. Features that cannot be matched end their track at the current timestep, and unmatched features in the next timestep start a new track.

Because the areal overlap is a more robust criterion than the distance between point positions, overlap tracking is particularly useful for fast-moving, crossing, or strongly deforming features. Optionally, the position of each feature can first be translated from the current to the next timestep using its tracked velocity (see {ref}`overlap-tracking-translate` below), which further improves the linking of fast-moving features.

## Basic usage

```python
import tobac

# detect features and also return the labeled feature mask
labels, features = tobac.feature_detection_multithreshold(
    data,
    dxy=dxy,
    threshold=[5, 10],
    position_threshold="weighted_abs",
    return_labels=True,
)

# link the features based on the overlap of their labeled masks
tracks = tobac.tracking.linking_overlap(features, labels)
```

The `mask` must be an `xarray.DataArray` with a `time` coordinate matching the `time` column of the features, and with dimensions `(time, hdim_1, hdim_2)` for 2D data or `(time, vdim, hdim_1, hdim_2)` for 3D data (see {ref}`overlap-tracking-3d` for 3D data with the vertical axis in a different position). Alternatively, a segmentation mask can be used:

```python
mask, features = tobac.segmentation.segmentation(features, data, dxy=dxy, threshold=5)
tracks = tobac.tracking.linking_overlap(features, mask)
```

## Parameters

A full list of parameters and descriptions can be found in the API reference: {py:func}`tobac.tracking.linking_overlap`. The parameters that affect how the overlap matching is performed are described below.

(overlap-tracking-overlap)=
### Minimum overlap

Two parameters control what counts as a valid overlap between a feature in one timestep and a feature in the next:

- `minimum_overlap` (int, default 1): minimum number of overlapping grid cells required for two features to be linked.
- `minimum_relative_overlap` (float, default 0): minimum fraction of overlap, computed as the number of overlapping grid cells divided by the size (number of grid cells) of the *smaller* of the two features. Must lie between 0 and 1. For example, `minimum_relative_overlap=0.5` requires at least half of the smaller feature to overlap with the other feature for a link to be made.

Increasing either parameter makes the linking stricter: partially overlapping features are then no longer linked, and instead end/start separate tracks.

(overlap-tracking-translate)=
### Predictive translation of feature positions

By default (`translate_method=None`), the overlap between timesteps is computed using the feature positions as detected. For fast-moving features it can help to first translate each feature from the current to the next timestep, so that the overlap is computed around the *predicted* position of the feature. This is controlled by `translate_method`:

- `"constant"`: each feature is translated by `velocity_constant` multiplied by the time difference between the two timesteps.
- `"drift"`: all features are translated by the *mean velocity* of all currently tracked features multiplied by the time difference, accounting for a uniform background drift of the whole field.
- `"predict"`: each feature is translated by its *own* tracked velocity multiplied by the time difference. The velocities are estimated from the matches in the previous timesteps; for the first two timesteps, velocities are bootstrapped from their overlap matches. For features without a tracked velocity (e.g., newly detected features), the missing velocities are filled according to `velocity_method`:
    - `"constant"`: use `velocity_constant`
    - `"mean"`: use the mean of the available tracked velocities
    - `"nearest"`: use the velocity of the nearest tracked feature (accounting for periodic boundaries, if set)
- `None` (default): no translation; the overlap is computed at the detected positions.

`velocity_constant` is given in grid cells per second and can be a single number or an array with one value per dimension, ordered like the position columns of the features (`hdim_1`, `hdim_2` for 2D data; `vdim`, `hdim_1`, `hdim_2` for 3D data). For example, with a time resolution of `dt = 600` s, `velocity_constant=np.array([8 / 600, 18 / 600])` translates each feature by 8 grid cells in `hdim_1` and 18 grid cells in `hdim_2` per timestep.

Example with predictive linking and a minimum relative overlap of 50%:

```python
tracks = tobac.tracking.linking_overlap(
    features,
    labels,
    minimum_relative_overlap=0.5,
    translate_method="predict",
    velocity_method="nearest",
)
```

### Gappy tracks: `memory`

`memory` (int, default 0) sets the number of timesteps a feature is allowed to vanish (i.e., to be undetected or unmatched) and still be linked to the same cell when it reappears. To this end, unmatched features from the last `memory` timesteps are kept as candidates for matching.

```{warning}
This parameter should be used with caution, as it can lead to erroneous trajectory linking, especially for data with low time resolution.
```

### Cell numbering and filtering

- `stubs` (int, default 1): minimum number of timesteps a cell must span to be kept in the output. Cells spanning fewer timesteps are marked as unassigned.
- `cell_number_start` (int, default 1): number assigned to the first tracked cell.
- `cell_number_unassigned` (int, default -1): value assigned to features that are not part of any tracked cell.

### Periodic boundary conditions

`PBC_flag` (`'none'`, `'hdim_1'`, `'hdim_2'`, or `'both'`; default `'none'`) specifies in which horizontal dimensions the domain is periodic. With periodic boundaries, translated feature positions wrap around the domain edges, and velocities between matched features are computed using the shortest distance across the boundary.

(overlap-tracking-3d)=
### 3D data

For 3D data, the `mask` has an additional vertical dimension and the features have a `vdim` column. The vertical axis of the mask is determined from `vertical_coord` (the name of the vertical coordinate, auto-detected if `None`), or can be given directly as an integer axis number via `vertical_axis`.

## Output

{py:func}`tobac.tracking.linking_overlap` returns the input features DataFrame with all original columns preserved, plus the same two columns added by the other tracking methods (see {doc}`tracking_output`):

- `cell`: integer identifier of the tracked cell each feature belongs to. Features that are not linked into any cell are assigned `cell_number_unassigned` (default -1).
- `time_cell`: time since the start of the cell, as a `pandas.Timedelta`; `NaT` for unassigned features.

## Examples

A worked example of overlap tracking, including the effect of the different parameters on the resulting tracks, can be found in the example gallery: [Methods and Parameters for Overlap Tracking](../../examples/Basics/Methods-and-Parameters-for-Overlap-Tracking/Methods-and-Parameters-for-Overlap-Tracking.html).

## See Also

- {py:func}`tobac.tracking.linking_trackpy` — centroid-based linking using trackpy
- {py:func}`tobac.tracking.linking_trackpy_latlon` — centroid-based linking for latitude/longitude data (see {doc}`lat_lon_tracking`)
- {doc}`tracking_output` — description of the tracking output variables
- {doc}`/getting_started/tracking_basics` — overview of tracking in the getting started guide
