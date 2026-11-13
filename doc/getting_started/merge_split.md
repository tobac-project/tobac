# Merge and Split

This submodule is a post processing step to address tracked cells which merge/split. 
The first iteration of this module is to combine the cells which are merging but have received a new cell id (and are considered a new cell) once merged. 
This module uses a minimum euclidian spanning tree to combine merging cells, thus the postfix for the function is MEST.
This submodule will label merged/split cells with a TRACK number in addition to its CELL number.

Features, cells, and tracks are combined using parent/child nomenclature. 
(quick note on terms; "feature" is a detected object at a single time step (see {doc}`feature_detection_overview`). "cell" is a series of features linked together over multiple timesteps (see {doc}`tracking_basics`). "track" may be an individual cell or series of cells which have merged and/or split.)

## Output formats: the `return_as` parameter

{py:func}`tobac.merge_split.merge_split_MEST` can return its output in one of two formats, selected with the `return_as` parameter:

* `return_as="dataset"` (default): returns an `xarray` dataset of tobac merge/split cells with parent and child designations, as described in the section below.
* `return_as="dataframe"`: returns the input `pandas` dataframe of tracked features with new columns containing the track, merge, and split information, as described in the dataframe output section at the end of this page.

The `track_number_unassigned` parameter sets the track value used for unassigned (untracked) features. It is only used when `return_as="dataframe"`, and if not given it defaults to the value of `cell_number_unassigned` (-1 by default).

Example usage:

```python
# xarray dataset of merge/split properties (default)
d = merge_split_MEST(Track)

# pandas dataframe: the input Track dataframe with track, merge and
# split information added as new columns
df = merge_split_MEST(Track, return_as="dataframe")
```

## Overview of the output dataset from merge_split (`return_as="dataset"`)

`d: xarray.core.dataset.Dataset`

xarray dataset of tobac merge/split cells with parent and child designations.

Parent/child variables include:

* `cell_parent_track_id`: The associated track id for each cell. All cells that have merged or split will have the same parent track id. If a cell never merges/splits, only one cell will have a particular track id.

* `feature_parent_cell_id`: The associated parent cell id for each feature. All feature in a given cell will have the same cell id.

* `feature_parent_track_id`: The associated parent track id for each feature. This is not the same as the cell id number.

* `track_child_cell_count`: The total number of features belonging to all child cells of a given track id.

* `cell_child_feature_count`: The total number of features for each cell.

* `cell_starts_with_split`: True for each cell that starts with a split from another cell.

* `cell_ends_with_merge`: True for each cell that ends with a merge into another cell.

merge_split outputs an `xarray` dataset with several variables. The variables, (with column names listed in the `Variable Name` column), are described below with units. Coordinates and dataset dimensions are Feature, Cell, and Track. 

Variables that are unique to merge/split output files:
```{csv-table} tobac Merge_Split Track Output Variables
:file: ./merge_split_out_vars.csv
:widths: 3, 35, 3, 3
:header-rows: 1
```

## Dataframe output (`return_as="dataframe"`)

When `return_as="dataframe"`, {py:func}`tobac.merge_split.merge_split_MEST` returns the input tracks dataframe with the track, merge and split information added as new columns. These columns correspond to the variables in the dataset output described above:

* `track`: The parent track id for each feature (the same as `feature_parent_track_id` in the dataset output). Features that are not part of a track are given the `track_number_unassigned` value.

* `time_track`: The time elapsed since the first feature of the track. Set to `NaT` for features that are not part of a track.

* `ends_with_merge`: True for the last feature of each cell that ends with a merge (corresponding to `cell_ends_with_merge` in the dataset output).

* `starts_with_split`: True for the first feature of each cell that starts with a split (corresponding to `cell_starts_with_split` in the dataset output).

The new columns added to the dataframe output are:
```{csv-table} tobac Merge_Split Dataframe Output Columns
:file: ./merge_split_dataframe_out_vars.csv
:widths: 3, 35, 3, 3
:header-rows: 1
```

The same columns can also be added to an existing merge/split dataset output with {py:func}`tobac.merge_split.add_track_info_to_dataframe`:

```python
d = merge_split_MEST(Track)  # dataset output
df = add_track_info_to_dataframe(Track, d)  # equivalent dataframe output
```
