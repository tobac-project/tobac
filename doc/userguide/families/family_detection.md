# Families

As described in Getting Started ({doc}`/getting_started/family_detection`), features ({doc}`/userguide/feature_detection/index`) are the *base unit* of *tobac*. **Families** stitch multiple *features* together. Families aggregate features, allowing one to link features together spatially. A family is defined as a contiguous region, at a single time, that is above or below the user-set {py:code}`threshold`. 

One can think about *families* as a kind of spatial clustering, although not using any preexisting clustering methods. As with every other method in *tobac*, family detection is modular, meaning that you can choose to use it or not or add on your own family detection methodology to *tobac*.

## Family Examples
- Identifying individual convective cores (features) within a broader MCS (family)
- Identifying individual updrafts (features) within a single cloud (family)
- Identifying individual thermals (features) within a cloud (family)

## Two Family Detection Methods

*tobac* has two functions for detecting families that use the same technique but on different data. Both are in {py:mod}`tobac.merge_split.families`.

### From data

{py:func}`tobac.merge_split.families.identify_feature_families_from_data` builds families from an input gridded field and a single {py:code}`threshold`. This can be the same field that you used for feature detection, but does not have to be, as long as they have the same dimensionality [^3dcaveat].

Similar to feature detection and segmentation, the {py:code}`target` parameter determines whether the families are defined by minima or maxima. Family detection also allows for you to create a *custom* mask (e.g., from a ML model or using multiple variables) and select it with {py:code}`target=bool`:

| {py:code}`target` | Family region |
|---|---|
| {py:code}`"maximum"` (default) | points > {py:code}`threshold` (e.g., radar reflectivity)|
| {py:code}`"minimum"` | points < {py:code}`threshold` (e.g., cold cloud tops in brightness temperature) |
| {py:code}`"bool"` | {py:code}`in_data` is already a {py:code}`True`/{py:code}`False` mask, {py:code}`True` is selected. {py:code}`threshold` is ignored |

Both comparisons are strict (i.e., using > or <, not ≥ or ≤): points equal to the threshold are *not* included.

### From segmentation

If you have already created a segmentation mask, {py:func}`tobac.merge_split.families.identify_feature_families_from_segmentation` builds families from the segmentation mask that you already have (see {doc}`/userguide/segmentation/index`). Any point assigned to a feature, other than {py:code}`unsegmented_point_values` (default {py:code}`0`) and {py:code}`below_threshold_values` (default {py:code}`-1`), counts as part of a family. Segmented areas from different features that touch each other are aggregated into one family. This is a good choice when you want families to match the segmented feature areas exactly, without choosing a second threshold.

## How the Algorithm Works

At each timestep, family detection does the following:

1. **Builds a mask.** As described above, a mask is generated. 
2. **Labels connected regions.** Each contiguous region in the mask gets a label. Points are connected only through shared faces (no diagonal connections). If you set {py:code}`PBC_flag` (`"hdim_1"`, `"hdim_2"`, or `"both"`), regions that cross a periodic boundary are joined.
3. **Assigns features to regions.**  Every feature that falls in a labeled region gets the same family ID.
4. **Drops regions without features.** Contiguous regions that contain no feature are not families. They are removed from the output grid (set to {py:code}`0`) and from the statistics.

Because sub-second time matching can be problematic, we allow a tolerance ({py:code}`time_padding`; default 0.5s) to match features to their columns in the gridded data.

### Things to keep in mind

- **A family is identified by its features' positions** A feature is only added to a family if its (rounded) position is inside the identified family region. If a feature's position falls outside every above/below-threshold region (e.g., if you set your family threshold more extreme than your feature detection threshold), it can't be linked to a family. These features get a family ID of {py:code}`unlinked_family_id`; default {py:code}`-1`. Set {py:code}`unlinked_family_id=None` to drop them from the output dataframe instead.

## Output

Both functions return:

1. **The feature dataframe with family IDs added.** This is the feature detection dataframe with a new column, set by {py:code}`family_column_name` (default `"feature_family_id"`), that holds each feature's family ID.
2. **A family statistics dataframe.** It has one row per family, indexed by family ID, with these columns:
   - `num_pixels`: the number of grid points in the family
   - `hdim_1_center`, `hdim_2_center`: the family's centroid in array-index coordinates. This is an unweighted mean of the family's grid points, and it does not account for periodic boundaries.
   - `time`: the time of the family
3. **(Optional) the family grid.** If {py:code}`return_grid=True`, you also get an {py:class}`xarray.DataArray` (named `family_grid`) with the same shape as the input. Each point holds its family ID, and {py:code}`0` marks points that are not in any family.

```python
from tobac.merge_split import families

# From data,
features_fam, family_stats, family_grid = families.identify_feature_families_from_data(
    features,
    cloud_water,
    threshold=1e-5,
    target="maximum",
    return_grid=True,
)

# From existing segmentation output
features_fam, family_stats = families.identify_feature_families_from_segmentation(
    features,
    segmentation_mask,
)
```


## Family Example Notebooks

```{nblinkgallery}
:caption: Jupyter Notebook Examples

../examples/Basics/Idealized-Family-Detection.ipynb
```


[^3dcaveat]: Although, you cannot currently project 2D features to 3D families.
