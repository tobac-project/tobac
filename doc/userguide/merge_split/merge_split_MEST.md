# Merging and Splitting Cells with `merge_split_MEST`

Tracking (see {doc}`/getting_started/tracking_basics`) links features over time into *cells*, and every cell receives a unique cell id. This means that when two tracked objects physically merge, the merged object continues under a **new** cell id, and when one object splits, each branch gets its own new cell id. {py:func}`tobac.merge_split.merge_split_MEST` is a post-processing step that finds these merge/split events and relabels the involved cells with a common **track** id, so that all cells belonging to one merged/split object share the same track number in addition to their individual cell numbers.

A full list of parameters and descriptions can be found in the API Reference: {py:func}`tobac.merge_split.merge_split_MEST`. The output formats (`return_as="dataset"` and `return_as="dataframe"`) and the output variables are described in {doc}`/getting_started/merge_split`.

## Basic Operating Procedure

For each tracked cell, `merge_split_MEST` takes the position of its **first** and **last** feature (converted from grid units to meters using `dxy`, and `dz` for 3D data). It then:

1. Finds all cell *start* points that lie within `distance` of each cell *end* point (a radius search; with periodic boundaries it uses a wrapped distance, see the Periodic Boundaries section below).
2. Builds a graph whose edges run from an ending cell to a starting cell, discarding edges to the same cell and edges whose frame gap is larger than `frame_len`.
3. Computes the minimum euclidean spanning tree (MEST) of that graph and takes its connected components: all cells connected through merge/split events receive the same parent track id.

Each surviving edge links the end of one cell to the start of a later, nearby cell. The cell whose end is linked is flagged `cell_ends_with_merge`, and the cell whose start is linked is flagged `cell_starts_with_split`. In the dataframe output, `ends_with_merge` is set on the last feature of each `cell_ends_with_merge` cell, and `starts_with_split` on the first feature of each `cell_starts_with_split` cell.

## Parameters and their effects

### `tracks` and `dxy`

`tracks` is the pandas DataFrame output from tracking (e.g. {py:func}`tobac.tracking.linking_trackpy`), and must contain the columns `feature`, `cell`, `frame`, `hdim_1`, and `hdim_2` (plus a vertical coordinate for 3D data). `dxy` is the horizontal grid spacing of the data **in meters**; it is required and is used to convert the `hdim_1`/`hdim_2` grid coordinates to physical distances, so it must match the grid spacing used in tracking.

### `distance`

`distance` is the distance threshold (in meters) determining how close the end of one cell and the start of another must be to be considered a merge/split. The default is `25 * dxy`. Increasing `distance` makes merge/split detection more aggressive: cells that end and begin farther apart are combined into the same track. Setting it too large can chain unrelated cells together through the spanning tree, so it should be chosen relative to how far cells can move between frames (compare with the search radius `d_max`/`v_max` used in tracking). Decreasing it makes the detection stricter and can miss genuine merges/splits of fast-moving objects.

### `frame_len`

`frame_len` (default: 5) is the maximum number of frames allowed between the end of one cell and the start of a related cell. Tracking can leave a gap of one or more frames between the last feature of a cell and the first feature of the cell it merged into or split from (e.g. when a feature temporarily drops below the detection threshold). Edges spanning a larger frame gap are discarded, so increasing `frame_len` allows connections across longer gaps, while decreasing it only connects cells that overlap or nearly overlap in time.

### `cell_number_unassigned`

`cell_number_unassigned` (default: -1) is the value that tracking assigned to features that are not part of any cell. Cells with this value are excluded from the merge/split analysis, and their features receive the unassigned track value in the output.

### Vertical coordinate: `dz` and `vertical_coord`

For 3D tracking output (a `vdim` column), the vertical positions are converted to meters as well. Either pass `dz`, the constant vertical grid spacing in meters (the coordinate is then assumed to be named `vdim`), or pass `vertical_coord`, the name of a vertical coordinate whose values are already in meters (if `None`, the coordinate is auto-detected from common names). Setting **both** `dz` and `vertical_coord` raises a `ValueError`, since the vertical spacing would be ambiguous.

### Periodic boundaries: `PBC_flag`, `min_h1`, `max_h1`, `min_h2`, `max_h2`

`PBC_flag` controls whether periodic boundary conditions are used when measuring distances between cell end and start points: `'none'` (default; `None` is treated the same) means no periodic boundaries, `'hdim_1'`/`'hdim_2'` mean the domain is periodic along that horizontal dimension, and `'both'` means it is periodic along both. If your domain is periodic (e.g. a global model grid), a cell can merge with another cell across the domain edge; without the matching `PBC_flag`, the wrapped distance across the boundary is never considered and such merges/splits are missed.

When periodic boundaries are used, the domain bounds must be given in grid units: `min_h1`/`max_h1` and `min_h2`/`max_h2` are the minimum and maximum (exclusive) points of the domain in `hdim_1` and `hdim_2`, with `max_h1 - min_h1` equal to the size of `hdim_1` (and likewise for `hdim_2`).

### Output format: `return_as` and `track_number_unassigned`

`return_as` selects the output format: `'dataset'` (default) returns an xarray Dataset of track/cell properties, while `'dataframe'` returns the input tracks DataFrame with the new `track`, `time_track`, `ends_with_merge`, and `starts_with_split` columns (see {doc}`/getting_started/merge_split`). `track_number_unassigned` sets the track value used for unassigned (untracked) features; it is only used with `return_as="dataframe"` and defaults to the value of `cell_number_unassigned`.

## Example

```python
import tobac

# xarray dataset of merge/split properties (default), with a 25 km 
# merge/split distance on a 1 km grid, with merges/splits found 
# for cells that end within 3 timesteps of each other
d = tobac.merge_split.merge_split_MEST(
    Track,
    dxy=1000,
    distance=25e3,
    frame_len=3,
)

# the same information as extra columns on the tracks dataframe
df = tobac.merge_split.merge_split_MEST(
    Track,
    dxy=1000,
    distance=25e3,
    frame_len=3,
    return_as="dataframe"
)
```
