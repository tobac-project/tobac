# Latitude/Longitude Tracking

In addition to the standard {py:func}`tobac.tracking.linking_trackpy` function, which tracks using
a Euclidean distance calculation that uses `dxy` to convert between array coordinates and physical
distances. The downside of this is that it requires the assumption of holding grid spacing
approximately constant, and for periodic boundaries (e.g., global tracking), it can be much slower
due to the overhead of calculating the wraparound boundaries. **Latitude/longitude tracking can be up to 10 times faster than typical tracking with periodic boundaries.**

Because tracking of geospatial data with latitude and longitude coordinates is so common in our
field, a specialized tracking function, {py:func}`tobac.tracking.linking_trackpy_latlon` is provided
to enable high-performance, grid spacing independent linking of features in 2D spatial fields that 
have latitude and longitude coordinates attached.

## When to use Lat/Lon Tracking

- You have 2D spatial data
- You have data that output a feature position from {py:func}`tobac.feature_detection.feature_detection_multithreshold` that contains columns that are for latitude and longitude
  - Note that you can either specify the names of these columns through `latitude_name` and 
    `longitude_name`, or you can let the function attempt to automatically detect the names.


## When not to use Lat/Lon Tracking

- You have data that is 3D spatially (i.e., there will be a `vdim` column in your output)
- You have data that is not on a latitude/longitude grid

## Examples using Lat/Lon Tracking

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card}
:link: ../../examples/Example_OLR_Tracking_model_latlon_tracking/Example_OLR_Tracking_model_latlon_tracking.html
:img-top: ../../_static/thumbnails/Example_OLR_Tracking_model_Thumbnail.png

Example OLR Tracking (lat/lon)
:::

::::