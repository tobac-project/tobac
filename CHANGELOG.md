### Tobac Changelog

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


**Bug fix**

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
