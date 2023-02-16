### Tobac Changelog

_**Version 1.4.2:**_


**Bug fix**

- Fixed a bug in the segmentation procedure that assigned the wrong grid cell areas to features in data frame  [#245](https://github.com/tobac-project/tobac/pull/245)

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
