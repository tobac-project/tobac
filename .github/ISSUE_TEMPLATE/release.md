---
name: Release
about: Prepare a new release for tobac
title: Release v.X.Y.Z
labels: release
assignees: ''

---

Checklist for releasing vX.Y.Z:

* [ ]  Re-run notebooks and commit updates to repository
* [ ]  Bump version in `__init__.py` in `RC_vX.Y.Z`
* [ ]  Add changelog in `RC_vX.Y.Z`
* [ ]  Add new contributors to vX.Y.Z
* [ ]  Merge `RC_vX.Y.Z` into `main`
* [ ]  Delete `RC_vX.Y.Z` branch
* [ ]  Create release
* [ ]  Push release to conda-forge
* [ ]  E-mail tobac mailing list
