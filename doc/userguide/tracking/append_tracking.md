# Appending New Features to Existing Tracks

## Overview

In some *tobac* workflows, you may need to *incrementally* grow your dataset (e.g., new data comes in or you want to break up your dataset into smaller chunks). You may have run {func}`tobac.tracking.linking_trackpy` on a batch of data and saved the tracks. You then get a new timestep (or timesteps) that you want to connect to the existing trajectories. Re-running the full tracking from scratch every time is expensive and can be confusing.

{func}`tobac.tracking.append_tracks_trackpy` enables, effectively, incremental tracking. It takes a previously tracked DataFrame (i.e., one that already contains a `cell` column) and a DataFrame of new untracked features, and extends the existing tracked features forward in time. {func}`tobac.tracking.append_tracks_trackpy` produces **identical** output (caveat: precise cell numbers may differ, but all tracks are identical) to running tracking on the full dataset simultaneously. Existing `cell` IDs are preserved.

Append tracking works with both `dxy`-based tracking from {func}`tobac.tracking.linking_trackpy` and latitude/longitude based tracking from {func}`tobac.tracking.linking_trackpy_latlon`. To append to tracks created with lat/lon tracking, set `use_latlon=True` (see {ref}`append-tracking-latlon` below).

---

## When to Use `append_tracks_trackpy`

Use {func}`tobac.tracking.append_tracks_trackpy` when **all** of the following are true:

- You have already called {func}`tobac.tracking.linking_trackpy` (or {func}`tobac.tracking.linking_trackpy_latlon`) on an earlier portion of your data and have a tracked DataFrame with a `cell` column.
- You have a new set of detected features from {py:func}`tobac.feature_detection.feature_detection_multithreshold`. Note: these features *can overlap* with the existing tracked feature DataFrame, but should not include tracked `cell` values. 

### Situations where you should *not* use append tracking

- **Inserting features into the middle of an existing track.** Only extending the track forward in time is supported.
- **Tracking with `memory > 0`.** This is not yet implemented and will raise a `NotImplementedError`.
- **3D data with lat/lon tracking.** Like {func}`tobac.tracking.linking_trackpy_latlon`, append tracking (`use_latlon=True`) only supports 2D data.

---

## Quick-Start Examples

### Minimal example

```python
import tobac

# --- Step 1: initial tracking run (as normal) ---
features_batch1 = tobac.feature_detection_multithreshold(
    data_batch1,
    dxy=dxy,
    threshold=[1, 2, 3],
)
tracks_batch1 = tobac.linking_trackpy(
    features=features_batch1,
    field_in=None,
    dt=dt,
    dxy=dxy,
    v_max=30,
    stubs=2,
    method_linking='predict',
)

# --- Step 2: new data arrives ---
features_batch2 = tobac.feature_detection_multithreshold(
    data_batch2,
    dxy=dxy,
    threshold=[1, 2, 3],
)

# --- Step 3: append without re-running from scratch ---
tracks_extended = tobac.tracking.append_tracks_trackpy(
    tracks_orig=tracks_batch1,
    new_features=features_batch2,
    dt=dt,
    dxy=dxy,
    v_max=30,
    stubs=2,
    method_linking='predict',
)
```

`tracks_extended` is a single DataFrame covering all timesteps from both batches. `cell` values assigned during `tracks_batch1` are unchanged.

---

### Appending multiple batches in a loop

If data arrives in a stream of batches (e.g. from a real-time feed), you can call `append_tracks_trackpy` repeatedly:

```python
import tobac

tracks = initial_tracks  # result of a prior linking_trackpy call

for new_data_chunk in data_stream:
    new_features = tobac.feature_detection_multithreshold(
        field_in=new_data_chunk,
        dxy=dxy,
        threshold=[1, 2, 3],
    )
    tracks = tobac.tracking.append_tracks_trackpy(
        tracks_orig=tracks,
        new_features=new_features,
        dt=dt,
        dxy=dxy,
        v_max=30,
        stubs=2,
        method_linking='predict',
    )

# tracks now covers the full time period
```

Each iteration extends `tracks` by one chunk. Because cell numbers only ever increase, all
previously assigned identifiers remain stable across iterations.

---

(append-tracking-latlon)=
### Appending to Lat/Lon Tracks

If your original tracks were created with {func}`tobac.tracking.linking_trackpy_latlon` (see {doc}`lat_lon_tracking`), pass `use_latlon=True` to {func}`tobac.tracking.append_tracks_trackpy`. Linking then uses the haversine distance between feature latitudes and longitudes, exactly as in {func}`tobac.tracking.linking_trackpy_latlon`, so the appended result is identical to lat/lon tracking of the full dataset at once.

```python
import tobac

# --- Step 1: initial lat/lon tracking run ---
tracks_batch1 = tobac.tracking.linking_trackpy_latlon(
    features=features_batch1,
    dt=dt,
    v_max=30,
    stubs=2,
    method_linking='predict',
)

# --- Step 2: append new features using lat/lon linking ---
tracks_extended = tobac.tracking.append_tracks_trackpy(
    tracks_orig=tracks_batch1,
    new_features=features_batch2,
    dt=dt,
    v_max=30,
    stubs=2,
    method_linking='predict',
    use_latlon=True,
)
```

When `use_latlon=True`, a few parameters are slightly different than when `use_latlon=False`, matching {func}`tobac.tracking.linking_trackpy_latlon`:

- `dxy` is not required and is ignored.
- `planet_radius` sets the radius (in meters) used in the haversine distance calculation. It defaults to an average Earth radius, but should be changed when using *tobac* on datasets from other planets.
- `d_min` is not supported; use `v_max` or `d_max` instead.
- For adaptive search, use `adaptive_step` together with `adaptive_stop_multiplier` instead of `adaptive_stop`. A `ValueError` will be raised if you use the wrong parameters.

---


## Common Errors and How to Fix Them

- **`ValueError: Need to have existing tracks.`**  
  - `tracks_orig` does not have a `cell` column. Make sure you are passing the output of
`linking_trackpy` (or a previously appended result), not a raw features DataFrame.

- **`NotImplementedError: Append tracks with memory not yet implemented.`**  
  - Set `memory=0` (the default). This is planned for a future feature.

- **`ValueError: One track is 3D, new track is 2D.`**  
  - Both `tracks_orig` and `new_features` must be either both 2D or both 3D. Check that feature detection was run with consistent settings for both batches.

- **`ValueError: Append tracks with use_latlon only supports 2D tracking.`**
  - Lat/lon append tracking only works for 2D features. For 3D data, use standard append tracking (`use_latlon=False`) with `dxy`.

- **`ValueError: dxy must be set unless use_latlon is True.`**
  - Either pass `dxy` for standard tracking, or set `use_latlon=True` if your original tracks came from {func}`tobac.tracking.linking_trackpy_latlon`.

- **`ValueError: PBC_flag must be 'none' with use_latlon ...`**, **`d_min is not supported with use_latlon ...`**, **`adaptive_stop is not supported with use_latlon ...`**, or **`adaptive_stop_multiplier is only supported with use_latlon ...`**
  - These parameters are specific to one tracking mode. See {ref}`append-tracking-latlon` for which parameters to use with `use_latlon=True`.

- **`ValueError: Error in appending tracks. Multiple pairs of cell:particle found.`**  
  - This is an internal consistency error that can occur. Fall back to a full `linking_trackpy` call. Please also  file a bug report on the tobac GitHub repository, as this should not occur. 

---

## See Also

- {func}`tobac.tracking.linking_trackpy` — the standard (non-append) trajectory linker
- {func}`tobac.tracking.linking_trackpy_latlon` — trajectory linker for lat/lon data
- {doc}`lat_lon_tracking` — user guide page on latitude/longitude tracking
