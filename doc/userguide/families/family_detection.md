# Families

As described in Getting Started ({doc}`/getting_started/family_detection`), features ({doc}`/userguide/feature_detection/index`) are the *base unit* of *tobac*, whereas **families** stitch multiple *features* together. Families aggregate features, allowing one to link features together spatially. 

One can think about *families* as a kind of spatial clustering, although at the moment not using any preexisting clustering methods. As with every other method in *tobac*, family detection is modular, meaning that you can choose to use it or not.  

## Family Examples
- Identifying individual convective cores (features) within a broader MCS (family)
- Identifying individual updrafts (features) within a single cloud (family)

## Algorithm Basics
You can either detect families from data ({py:func}`tobac.merge_split.families.identify_feature_families_from_data`) or from segmentation output ({py:func}`tobac.merge_split.families.identify_feature_families_from_segmentation`). Similar to segmentation, families are identified based on a single threshold when detected from data (defined by {py:code}`threshold`), but it does *not* need to be the same field that you detect the features on[^3dcaveat].

## Family Example Notebooks

```{nblinkgallery}
:caption: Jupyter Notebook Examples

../examples/Basics/Idealized-Family-Detection.ipynb
```


[^3dcaveat]: Although, you cannot currently project 2D features to 3D families.