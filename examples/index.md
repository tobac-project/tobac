# Example Gallery

tobac is provided with a set of Jupyter notebooks that show examples of the application of tobac for different types of datasets.


## Fundamentals of Detection and Tracking
```{nbgallery}
   :caption: Fundamentals of Detection and Tracking

   Test Blob in 2D <./Basics/Idealized-Case-1_Tracking-of-a-Test-Blob-in-2D>
   Two crossing Blobs <./Basics/Idealized-Case-2_Two_crossing_Blobs.ipynb>

   On Feature Detection: Part 1 <./Basics/Methods-and-Parameters-for-Feature-Detection_Part_1.ipynb>
   On Feature Detection: Part 2 <./Basics/Methods-and-Parameters-for-Feature-Detection_Part_2.ipynb>
   On Segmentation <./Basics/Methods-and-Parameters-for-Segmentation.ipynb>
   On Linking <./Basics/Methods-and-Parameters-for-Linking.ipynb>
```

## Examples of Using *tobac* with Observations
```{nbgallery}
   :caption: Examples of Using tobac with Observations

   OLR from GOES-13 Satellite <./Example_OLR_Tracking_satellite/Example_OLR_Tracking_satellite>
   Combine Radar & Satellite <./Example_Track_on_Radar_Segment_on_Satellite/Example_Track_on_Radar_Segment_on_Satellite>
```

## Examples of Using *tobac* with Model Data
```{nbgallery}
   :caption: Examples of Using tobac with Model Data

   WRF OLR <./Example_OLR_Tracking_model/Example_OLR_Tracking_model>
   WRF Precip <./Example_Precip_Tracking/Example_Precip_Tracking>
   WRF Updrafts <./Example_Updraft_Tracking/Example_Updraft_Tracking>
   WRF Mesoscale Vorticity <./Example_vorticity_tracking_model/Example_vorticity_tracking_model>
   DALES Cloud Botany LWP (cumulus tracking) <./Example_low_cloud_tracking_eurec4a/Example_low_cloud_tracking_eurec4a>
   ICON Global MCS Tracking <./Example_ICON_MCS_tracking/Example_ICON_MCS_tracking>
```

## Calculating Bulk Statistics
```{nbgallery}
   :caption: Calculating Bulk Statistics

   Calculating Bulk Statistics during Feature Detection </userguide/bulk_statistics/notebooks/compute_statistics_during_feature_detection>
   Calculating Bulk Statistics during Segmentation </userguide/bulk_statistics/notebooks/compute_statistics_during_segmentation>
   Calculating Bulk Statistics as a Postprocessing Step </userguide/bulk_statistics/notebooks/compute_statistics_postprocessing_example>
```

## Examples of Using *tobac* with Large Datasets and in Parallel
```{nbgallery}
   :caption: Examples of Using tobac with Large Datasets and in Parallel

   Simple Dask Parallel Tutorial <./big_data_processing/parallel_processing_tobac>
```

The notebooks can be found in the **examples** folder as part of the python package. The necessary input data for these examples is avaliable on zenodo and can be downloaded automatically by the Jupyter notebooks.