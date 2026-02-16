# Families

Features ({doc}`/userguide/feature_detection/index`) are the *base unit* of *tobac*. However, sometimes you want to identify how features interact together. **Families** are the way to do this. Families aggregate features, allowing one to link features together spatially. 

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