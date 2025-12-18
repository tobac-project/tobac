### Tobac Changelog

_**Version 1.6.2:**_

**Enhancements for Users**

- Updated Python version requirements to Python 3.9-3.13. This fixes bugs with Python 3.14 issues upstream, and allows us to introduce thorough typechecking (Python >=3.9) in *tobac*. [#532](https://github.com/tobac-project/tobac/pull/532), [#541](https://github.com/tobac-project/tobac/pull/541/)
- Added support for calculating 3D velocity components. [#530](https://github.com/tobac-project/tobac/pull/530)
- Added support for returning the detected field from feature detection [#488](https://github.com/tobac-project/tobac/pull/488)

**Documentation**

- The documentation pages have been overhauled to use a new, modern theme and have had several pages added to them. [#477](https://github.com/tobac-project/tobac/pull/477)
- Multiple new examples were added to the documentation [#486](https://github.com/tobac-project/tobac/pull/486), [#492](https://github.com/tobac-project/tobac/pull/492), [#500](https://github.com/tobac-project/tobac/pull/500)

**Internal Enhancements**

- Updated pyproject.toml and removed setup.py to support eventual release on PyPi [#537](https://github.com/tobac-project/tobac/pull/537)


_**Version 1.6.1:**_

**Bug fixes**

- An bug in the `field_and_features_over_time` generator, used to loop through mask `DataArrays` and feature `Dataframes` simultaneously which caused an error in the dataframe had no 0 index value has been fixed [#506](https://github.com/tobac-project/tobac/pull/506)
- Interpolation of non-numeric coordinates (such as datetime) now works, and chooses the nearest element [#509](https://github.com/tobac-project/tobac/pull/509)

_**Version 1.6.0:**_

**Enhancements for Users**

- MAJOR CHANGE: Feature detection and segmentation have been changed to use xarray DataArrays for representation of spatial fields instead of Iris Cubes. This change will allow for increased flexibility for input data and ease development. Support for Iris Cube input is maintained through a conversion wrapper. [#354](https://github.com/tobac-project/tobac/pull/354) & [#417](https://github.com/tobac-project/tobac/pull/417)
- Some differences in the output of feature detection are present in the output from feature detection: when using xarray, the column names for coordinates will use the variable names, while with iris the columns will use the standard names. This differences reflects the different ways in which coordinates are named in xarray and iris, and can be overridden by the user using the `use_standard_names` parameter. [#489](https://github.com/tobac-project/tobac/pull/489)

**Bug fixes**

- An issue with datetime output from feature detection being converted to `cftime` format erroneously was fixed, and datetime formats should now match the formatting in the input field. Time matching within a tolerance range now works with any combination of datetime formats [#489](https://github.com/tobac-project/tobac/pull/489)

**Documentation**

- Example notebooks have been converted to use `xarray`, and have had minor enhancements to descriptions and visualisations. A notebook showing the legacy approach using Iris Cube input is retained [#487](https://github.com/tobac-project/tobac/pull/487)

**Internal Enhancements**

- `tobac.utils.internal.basic` was restructured into `coordinates` and `label_props` utils, and new `datetime` and `generators` utils were added [#489](https://github.com/tobac-project/tobac/pull/489)

_**Version 1.5.5:**_

**Bug fixes**

- Including of annotations import for python versions before 3.10 [#468](https://github.com/tobac-project/tobac/pull/468)
- Fix bulk statistics calculation when provided a dask array [#474](https://github.com/tobac-project/tobac/pull/474)

**Internal Enhancements**

- Fix matrix testing to use the specified python versions [#468](https://github.com/tobac-project/tobac/pull/468)


_**Version 1.5.4:**_

**Enhancements for Users**

- Added the ability to use the Minimum Euclidean Spanning Tree merge/split method on data with periodic boundaries [#372](https://github.com/tobac-project/tobac/pull/372)
- Added the ability to calculate online bulk statistics during feature detection on the raw (i.e., unsmoothed) data [#449](https://github.com/tobac-project/tobac/pull/449)

**Bug fixes**

- Fixes to calculations of bulk statistics [#437](https://github.com/tobac-project/tobac/pull/437)
- Fixes to handling of PBC feature points on the PBC wraparound border [#434](https://github.com/tobac-project/tobac/pull/434)
- Fixed an error that allows non-matching features to be used in the offline bulk statistics calculation [#448](https://github.com/tobac-project/tobac/pull/448)
- Fixed a bug that prevented using minimum distance filtering with varying vertical coordinates [#452](https://github.com/tobac-project/tobac/pull/452)

**Documentation**

- Add thumbnails to the new example gallery [#428](https://github.com/tobac-project/tobac/pull/428)
- Added documentation for developers [#281](https://github.com/tobac-project/tobac/pull/281)
- Updated documentation for the `n_min_threshold` function in feature detection [#432](https://github.com/tobac-project/tobac/pull/432)
- Added documentation for dealing with big datasets [#408](https://github.com/tobac-project/tobac/pull/408)
- Updated documentation to note that the *tobac* v1.5.0 paper in GMD is in its final form [#450](https://github.com/tobac-project/tobac/pull/450) 

**Internal Enhancements**

- PBC Distance Function handling improved for tracking and other portions of the library that uses it [#386](https://github.com/tobac-project/tobac/pull/386)
- Added tests to `tobac.utils.get_spacings`  [#429](https://github.com/tobac-project/tobac/pull/429)
- Added matrix testing for Python 3.12  [#451](https://github.com/tobac-project/tobac/pull/451)
- Resolved issues around updating dependencies in `black` formatting checks and Zenodo JSON checks [#457](https://github.com/tobac-project/tobac/pull/457)


_**Version 1.5.3:**_

**Enhancements for Users**

- Update `calculate_area` to allow the calculation of the projected 2D area of 3D objects, and enhance bulk statistics to allow calculation of statistics on the projected footprint on 2D fields [#378](https://github.com/tobac-project/tobac/pull/378)

**Bug fixes**

- Fix a bug in `get_spacing` that would return a negative value if one coordinate was in ascending order and the other in descending order, and fix other bugs where the wrong coordinate was referenced [#400](https://github.com/tobac-project/tobac/pull/400)

**Documentation**

- Re-integration of notebooks from the tobac-tutorials repo [#334](https://github.com/tobac-project/tobac/pull/334)
- Add example gallery to readthedocs page [#411](https://github.com/tobac-project/tobac/pull/411)

**Internal Enhancements**

- Add ability to save whether iris to xarray conversion ocurred and update decorators to allow keyword parameters [#380](https://github.com/tobac-project/tobac/pull/380)
- Reorganisation of analysis tools into analysis package [#378](https://github.com/tobac-project/tobac/pull/378)


_**Version 1.5.2:**_

**Enhancements for Users**

- Let users optionally derive bulk statistics of the data points belonging to each feature. Bulk statistics can be calulcated during feature detection, segmentation or afterwards by applying one of more functions to each feature [#293](https://github.com/tobac-project/tobac/pull/293)
- Wrapped functions now show the correct docstring [#359](https://github.com/tobac-project/tobac/pull/359)

**Bug fixes**

- Fixed an out-of-bounds error that could occur when performing segmentation with PBCs [#350](https://github.com/tobac-project/tobac/pull/350)
- Path to data in example notebooks fixed after changes to zenodo [#357](https://github.com/tobac-project/tobac/pull/357)
- Bulk statistics updated to use multiple fields correctly and perform numpy-style broadcasting [#368](https://github.com/tobac-project/tobac/pull/368)
- PBCs now work when using predictive tracking [#376](https://github.com/tobac-project/tobac/pull/376)
- Fixed error with PBC distance calculation using numba if all of `min_h1`, `max_h1`, `min_h2`, `max_h2` were not specified, even if not calculating PBCs over one of the dimensions [#384](https://github.com/tobac-project/tobac/pull/384)

**Documentation**

- Documentation on use of `time_cell_min` in tracking added [#361](https://github.com/tobac-project/tobac/pull/361)
- Documentation on how thresholding is applied in segmentation update [#365](https://github.com/tobac-project/tobac/pull/365)
- References to tobac papers updated [#365](https://github.com/tobac-project/tobac/pull/365)

**Internal Enhancements**

- Type hints added for feature detection [#337](https://github.com/tobac-project/tobac/pull/337)
- Reorganisation and addition of type hints to interal utils [#241](https://github.com/tobac-project/tobac/pull/241)
- Type hints added for segmentation [#351](https://github.com/tobac-project/tobac/pull/351)

**Repository Enhancements**

- Matrix CI testing added for multiple python versions on Linux, MacOS and Windows [#353](https://github.com/tobac-project/tobac/pull/353)
- Issue templates with checklists added [#358](https://github.com/tobac-project/tobac/pull/358)
- Black reformatting updated to say what is wrong [#362](https://github.com/tobac-project/tobac/pull/362)
- Pylint CI workflow added to assess code quality and compare to base branch [#373](https://github.com/tobac-project/tobac/pull/373)


_**Version 1.5.1:**_

**Bug fixes**

- The `strict_thresholding` option in feature detection now works correctly for detecting minima, and produces the same results as without strict thresholding if the `n_min_threshold` is a scalar value [#316](https://github.com/tobac-project/tobac/pull/316)
- utils.general.standardize_track_dataset was added back after being inadvertently removed in version 1.5.0 [#330](https://github.com/tobac-project/tobac/pull/330)
- All Numba import errors are now caught with the exception of KeyboardInterrupts. [#335](https://github.com/tobac-project/tobac/pull/335)

**Documentation**
- Fix to readthedocs building after system packages no longer imported [#336](https://github.com/tobac-project/tobac/pull/336)

**Repository Enhancements**
- Fix to Jupyter Notebook CI that was timing out due to installing packages with `conda`, switched to `mamba` to resolve. [#340](https://github.com/tobac-project/tobac/pull/340)


_**Version 1.5.0:**_

**Enhancements for Users**

- Feature detection and tracking in three dimensions is now supported [#209](https://github.com/tobac-project/tobac/pull/209)
- Feature detection, segmentation, and tracking across periodic boundaries is now supported [#259](https://github.com/tobac-project/tobac/pull/259)
- Transformation of feature detection points to allow segmentation on a new grid is now supported [#242](https://github.com/tobac-project/tobac/pull/242)
- `n_min_threshold` in feature detection can now be set for each threshold level instead of uniformly for all thresholds [#208](https://github.com/tobac-project/tobac/pull/208)
- Feature detection now has the option to only detect a feature if all previous thresholds have been met [#283](https://github.com/tobac-project/tobac/pull/283)
- Unsegmented points can now have their marker value selected [#285](https://github.com/tobac-project/tobac/pull/285)
- Minimum distance filtering is now substantially faster [#249](https://github.com/tobac-project/tobac/pull/249)
- `combine_feature_dataframes` now allows the retention of feature numbers [#300](https://github.com/tobac-project/tobac/pull/300)
- `scikit-learn` is now a required dependency; `pytables` and `cf-units` are no longer direct dependencies of *tobac* [#204](https://github.com/tobac-project/tobac/pull/204)

**Bug fixes**

- An error is now raised if none of the search range parameters (`v_max`, `d_max`, `d_min`) are set in `linking_trackpy` [#223](https://github.com/tobac-project/tobac/pull/223)
- Minimum distance filtering in feature detection (set through `min_distance` in `feature_detection_multithreshold`, run through `filter_min_distance`) has been fixed to properly work when `target=minimum`. [#244](https://github.com/tobac-project/tobac/pull/244)
- Interpolated numeric coordinates now preserve their datatypes (i.e., `float`s stay `float`s) in feature detection [#250](https://github.com/tobac-project/tobac/pull/250)
- Fixes to the internal `find_axis_from_coord` utility to allow for non-dimensional coordinates to be correctly dealt with [#255](https://github.com/tobac-project/tobac/pull/255)
- Jupyter notebooks changed to use string paths to work around an Iris bug [#294](https://github.com/tobac-project/tobac/pull/294)
- Minimum distance filtering updated to produce consistent results [#249](https://github.com/tobac-project/tobac/pull/249)

**Documentation**

- Enhancements to the documentation of how *tobac* links features together [210](https://github.com/tobac-project/tobac/pull/210)
- Fixes to the API documentation generation when using type hints [#305](https://github.com/tobac-project/tobac/pull/305)

**Internal Enhancements**

- New converting decorators (`xarray_to_iris`, `iris_to_xarray`, `xarray_to_irispandas`, `irispandas_to_xarray`) have been added for internal use that will allow the upcoming transition to xarray throughout *tobac* to occur more smoothly. [#179](https://github.com/tobac-project/tobac/pull/179)
- The `utils` module has been broken up from a single `utils.py` file to multiple files inside a `utils` folder, allowing for ease of maintenance and fewer long code files. [#191](https://github.com/tobac-project/tobac/pull/191)
- `scipy.interpolate.interp2d` in `add_coordinates` and `add_coordinates_3D` has been replaced with `scipy.interpolate.interpn` as `interp2d` has been deprecated. [#279](https://github.com/tobac-project/tobac/pull/279) 
- `setup.py` updated to draw its required packages from `requirements.txt` [#288](https://github.com/tobac-project/tobac/pull/288)



**Repository Enhancements**

- New CI was added to automatically check the example and documentation Jupyter notebooks [#258](https://github.com/tobac-project/tobac/pull/258), [#290](https://github.com/tobac-project/tobac/pull/290)
- The `check_formatting` CI action has been revised to install dependencies through `conda` [#288](https://github.com/tobac-project/tobac/pull/288)
- Repository authors updated [#289](https://github.com/tobac-project/tobac/pull/289)
- CI added to check author list formatting for Zenodo [#292](https://github.com/tobac-project/tobac/pull/292)


**Deprecations**

- All functions in `centerofgravity.py` (`calculate_cog`, `calculate_cog_untracked`, `center_of_gravity`) have been deprecated and will be removed or significantly changed in v2.0. [#200](https://github.com/tobac-project/tobac/pull/200)
- `plot_mask_cell_track_follow`, `plot_mask_cell_individual_follow`, `plot_mask_cell_track_static`, `plot_mask_cell_individual_static`, `plot_mask_cell_track_2D3Dstatic`, `plot_mask_cell_track_3Dstatic`, `plot_mask_cell_individual_3Dstatic`, and `plot_mask_cell_track_static_timeseries` in `plotting.py` have been deprecated and will be removed or significantly changed in v2.0. [#200](https://github.com/tobac-project/tobac/pull/200)
- The wrapper functions in `wrapper.py` (`tracking_wrapper` and `maketrack`) have been deprecated and will be removed in v2.0. [#200](https://github.com/tobac-project/tobac/pull/200)
- `cell_statistics_all`, `cell_statistics`, and `cog_cell` in the analysis module have been deprecated and will be removed or significantly changed in v2.0. [#207](https://github.com/tobac-project/tobac/pull/207)
- `tobac.utils.combine_tobac_feats` has been renamed to `tobac.utils.combine_feature_dataframes`, and the original name has been deprecated and will be removed in a future release. [#300](https://github.com/tobac-project/tobac/pull/300)



_**Version 1.4.2:**_


**Bug fixes**

- Fixed a bug in the segmentation procedure that assigned the wrong grid cell areas to features in data frame  [#246](https://github.com/tobac-project/tobac/pull/246)

- Fixed a bug in feature_detection.filter_min_distance() that always selected the feature with the largest threshold, even if the feature detection is targeting minima. The target is now an optional input parameter for the distance filtering [#251](https://github.com/tobac-project/tobac/pull/251)

- Fixed an issue in the 2D coordinate interpolation that produced object dtypes in feature detection and made the feature input data frame incompatible with the merge and split function [#251](https://github.com/tobac-project/tobac/pull/251)


_**Version 1.4.1:**_

**Bug fixes**

- Fixed a bug with predictive tracking that would duplicate column names if the input dataset has coordinates x and/or y [#217](https://github.com/tobac-project/tobac/pull/217)
- Set extrapolate parameter to 0 in example notebooks to prevent not implemented error [#217](https://github.com/tobac-project/tobac/pull/217)

**Documentation**

- Regenerated example notebooks so that they are up to date for the present version [#233](https://github.com/tobac-project/tobac/pull/233)

_**Version 1.4.0:**_

**Enhancements**

- Added the ability to detect feature mergers and splits ([#136](https://github.com/tobac-project/tobac/pull/136))
- Added spectral filtering of input data to feature detection (#137)
- Substantial improvements to documentation ([#138](https://github.com/tobac-project/tobac/pull/138), [#150](https://github.com/tobac-project/tobac/pull/150), [#155](https://github.com/tobac-project/tobac/pull/155), [#173](https://github.com/tobac-project/tobac/pull/173), [#189](https://github.com/tobac-project/tobac/pull/189), [#195](https://github.com/tobac-project/tobac/pull/195), [#197](https://github.com/tobac-project/tobac/pull/197))
- Added a new function to combine feature dataframes when feature detection is run in parallel ([#186](https://github.com/tobac-project/tobac/pull/186))

**Bug fixes**

- Reset the adaptive search parameters back to default when using adaptive trackpy tracking ([#168](https://github.com/tobac-project/tobac/pull/168))
- Added checks to make sure that both `adaptive_step` and `adaptive_stop` are set when using adaptive tracking ([#168](https://github.com/tobac-project/tobac/pull/168))
- Added error raising when trying to use the not yet implemented extrapolation feature ([#177](https://github.com/tobac-project/tobac/pull/177))
- Fixed a bug where `min_distance` did not work in feature detection ([#187](https://github.com/tobac-project/tobac/pull/187))
- Fixed a bug where feature detection output different feature locations depending on the order that thresholds are passed in ([#199](https://github.com/tobac-project/tobac/pull/199))

**Documentation**

- Updated docstrings to NumPy format ([#138](https://github.com/tobac-project/tobac/pull/138), [#155](https://github.com/tobac-project/tobac/pull/155), [#173](https://github.com/tobac-project/tobac/pull/173))
- Enabled API documentation generation ([#150](https://github.com/tobac-project/tobac/pull/150))
- Enhanced documentation of feature detection and segmentation parameters ([#150](https://github.com/tobac-project/tobac/pull/150))
- Added contributors to zenodo ([#139](https://github.com/tobac-project/tobac/pull/139))
- Added `__version__` as a parameter ([#175](https://github.com/tobac-project/tobac/pull/175))
- Updated the feature detection docstrings to add clarification around units ([#189](https://github.com/tobac-project/tobac/pull/189))
- Added documentation on why sometimes no features are segmented ([#195](https://github.com/tobac-project/tobac/pull/195))
- Added updates to README file, including linking the google groups ([#162](https://github.com/tobac-project/tobac/pull/162), [#197](https://github.com/tobac-project/tobac/pull/197))

**Repository enhancements**

- Specified the version of `black` to use for validating formatting during CI ([#161](https://github.com/tobac-project/tobac/pull/161))
- Lowered threshold before code coverage CI fails on pull requests ([#159](https://github.com/tobac-project/tobac/pull/159))

**Deprecations**

- Support for Python 3.6 and earlier is now deprecated and will be removed in v1.5.0 ([#193](https://github.com/tobac-project/tobac/pull/193))

_**Version 1.3.3:**_

**Bug fixes**

- Added a workaround to a bug in trackpy that fixes predictive tracking [#170](https://github.com/tobac-project/tobac/pull/170)

_**Version 1.3.2:**_

**Bug fixes**

- Fixed a bug with Feature Detection that caused it to fail when using `weighted_abs` position [#148](https://github.com/tobac-project/tobac/pull/148)
- Fixed a bug where adaptive_search within `linking_trackpy` was not working correctly [#140](https://github.com/tobac-project/tobac/pull/140)

**Repository enhancements**

- Added automatic code coverage reports [#124](https://github.com/tobac-project/tobac/pull/124)
- Added automatic building of readthedocs documentation on pull requests

_**Version 1.3.1:**_

**Enhancements**

- Added auto-downloading of files in the example notebooks if data isn't already present [#113](https://github.com/tobac-project/tobac/pull/113)

**Bug fixes**

- Fixed a bug with `map_tracks` that had it plot untracked cells caused by the switch to `-1` for untracked cells [#130](https://github.com/tobac-project/tobac/pull/130)

**Repository enhancements**

- New pull request template for the repository, including a checklist to be completed for each pull request [#120](https://github.com/tobac-project/tobac/pull/120)

_**Version 1.3:**_

**Enhancements**

- Significant performance improvements for tracking [#89](https://github.com/climate-processes/tobac/pull/89)
- Significant performance improvements for feature detection and segmentation [#90](https://github.com/climate-processes/tobac/pull/90)
- Performance improvement for `calculate_area` [#93](https://github.com/climate-processes/tobac/issues/93)
- Added ability to set a user defined stub cell value instead of `np.nan`. Default value is `-1` and stub cell values are now integers instead of floats by default [#74](https://github.com/climate-processes/tobac/issues/93)
- Added deprecation warnings for parameters `min_num` in feature detection and `d_min` in tracking, and added exceptions when multiple, incompatible parameters are given (e.g. `d_max` and `v_max`) [#107](https://github.com/climate-processes/tobac/pull/107)

**Bug fixes**

- Fixed level parameter in segmentation, as this previously had no effect [#92](https://github.com/climate-processes/tobac/pull/92)
- Remove `is` comparisons for string literals [#99](https://github.com/climate-processes/tobac/pull/99)
- Added missing `raise` for exception in `get_spacings` [#105](https://github.com/climate-processes/tobac/pull/105)
- Remove automatic setting of matplotlib backend to `agg` on import [#100](https://github.com/climate-processes/tobac/pull/100)
- Fix deprecation warnings for changed import paths in dependencies [#110](https://github.com/climate-processes/tobac/pull/110)

**Documentation**

- Added recommended python style [#72](https://github.com/climate-processes/tobac/issues/72)
- Updated author list and email addresses [#109](https://github.com/climate-processes/tobac/pull/109)

**Repository enhancements**

- Black formatting of all python code and formatting check in actions [#78](https://github.com/climate-processes/tobac/pull/78)
- Pre-commit hook for black formatting [#96](https://github.com/climate-processes/tobac/pull/96)
