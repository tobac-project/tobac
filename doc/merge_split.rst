*Merge and Split 
======================


This submodule is a post processing step to address tracked cells which merge/split. 
The first iteration of this module is to combine the cells which are merging but have received a new cell id (and are considered a new cell) once merged. 
This submodule will label merged/split cells with a TRACK number in addition to its CELL number.

Features, cells, and tracks are combined using parent/child nomenclature. 
(quick note on terms; “feature” is a detected object at a single time step. “cell” is a series of features linked together over multiple timesteps. "track" may be an individual cell or series of cells which have merged and/or split.)
*Merge and Split 
======================


This submodule is a post processing step to address tracked cells which merge/split. 
The first iteration of this module is to combine the cells which are merging but have received a new cell id (and are considered a new cell) once merged. 
This submodule will label merged/split cells with a TRACK number in addition to its CELL number.

Features, cells, and tracks are combined using parent/child nomenclature. 
(quick note on terms; “feature” is a detected object at a single time step. “cell” is a series of features linked together over multiple timesteps. "track" may be an individual cell or series of cells which have merged and/or split.)


Overview of the output dataframe from merge_split
d : xarray.core.dataset.Dataset
    xarray dataset of tobac merge/split cells with parent and child designations.

Parent/child variables include:
- cell_parent_track_id: The associated track id for each cell. All cells that have merged or split will have the same parent track id. If a cell never merges/splits, only one cell will have a particular track id.
- feature_parent_cell_id: The associated parent cell id for each feature. All feature in a given cell will have the same cell id.
- feature_parent_track_id: The associated parent track id for each feature. This is not the same as the cell id number.
- track_child_cell_count: The total number of features belonging to all child cells of a given track id.
- cell_child_feature_count: The total number of features for each cell.


Example usage:
        
d = merge_split(Track)


Overview of the output dataframe from merge_split
d : xarray.core.dataset.Dataset
    xarray dataset of tobac merge/split cells with parent and child designations.

Parent/child variables include:
- cell_parent_track_id: The associated track id for each cell. All cells that have merged or split will have the same parent track id. If a cell never merges/splits, only one cell will have a particular track id.
- feature_parent_cell_id: The associated parent cell id for each feature. All feature in a given cell will have the same cell id.
- feature_parent_track_id: The associated parent track id for each feature. This is not the same as the cell id number.
- track_child_cell_count: The total number of features belonging to all child cells of a given track id.
- cell_child_feature_count: The total number of features for each cell.


Example usage:
        
d = merge_split(Track)


merge_split outputs an `xarray` dataset with several variables. The variables, (with column names listed in the `Variable Name` column), are described below with units. Coordinates and dataset dimensions are Feature, Cell, and Track. 

Variables that are common to all feature detection files:

.. csv-table:: tobac Merge_Split Track Output Variables
   :file: ./merge_split_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

