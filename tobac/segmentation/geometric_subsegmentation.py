#!/usr/bin/env python

# Copyright (c) 2012- 2017

# TROPOS,
# Permoserstr. 15
# 04318 Leipzig, Germany.

# Author:
# ======
# Fabian Senf <senf@tropos.de>


# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of
# the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


"""
Package to perform segmentation of fields.

The special emphasis is on threshold-based segmentation of
2-dim atmoshperic fields for which object-based analysis
techniques should be applied.

**Notes**

* the package has variants of connected compund analsis & watershedding
* some split-and-merge rules exists
* most of the method might be also applied to 3-dim fields
"""

# load libraries -----------------------------------------------------
import sys, os, glob

import numpy as np
import datetime

import scipy.ndimage
import skimage.feature
import skimage.measure

try:
    import mahotas as mh
except ImportError:
    print("mahotas package not available - do not use mahotas options")

try:
    from skimage.segmentation import watershed
except ImportError:
    from skimage.morphology import watershed

from tobac.utils.general import curve_flow_filter

######################################################################
# (1) Single Threshold Clustering
######################################################################


def clustering(f, thresh, cluster_method="watershed", min_size=20, **kwargs):
    """
    General interface to aplly either connectivity or watershed clustering.


    Parameters
    ----------
    f : np.array
        field to be segmented

    thresh : float
        threshold for masking, set foreground to f > thresh

    cluster_method : {'watershed', 'watershed_merge', 'connect'}, optional
        switch between watershed or connectivity clustering

    min_size : int, optional, default = 20
        minimum cluster size (px) left in the cluster field

    **kwargs : dict, optional
        keyword argument for selected cluster algorithm



    Returns
    -------
    c : np.array
        categorial cluster field
    """

    if cluster_method == "watershed":
        c = watershed_clustering(f, thresh, **kwargs)
        return remove_small_clusters(c, min_size=min_size)

    elif cluster_method == "geometric_subsegmentation":
        return watershed_merge_clustering(f, thresh, min_size=min_size, **kwargs)

    elif cluster_method == "connect":
        c = connectivity_clustering(f, thresh, **kwargs)
        return remove_small_clusters(c, min_size=min_size)

    else:
        raise KeyError("Unkown cluster method")


######################################################################
######################################################################


def watershed_clustering(
    f,
    thresh,
    dradius=3,
    ctype=4,
    marker_field="dist",
    filter_method="curve",
    numberOfIterations=5,
    cluster_masking=True,
    exclude_border=False,
    marker_method="mahotas_regmax",
    siggauss=1.0,
    marker_shrinkage_Niter=3,
    marker_shrinkage_dmin=0.1,
    **kwargs,
):
    """
    Applies watershed clustering to a field.


    Parameters
    ----------
    f : np.array
        field to be segmented

    thresh : float
        threshold for masking, set foreground to f > thresh

    dradius : int, optional
        minimal distance for neighboring maxima

    ctype: {4, 8}, optional
        set either 4- or 8-connectivity (edge vs. edges+corners)

    marker_field : str, optional
        switch if input field `f` or a distance transform as used
        to set the watershed markers

        if 'dist', then max. in Euclidean distance to background
        is used to set initial markers, else the max. in field f
        are taken

    filter_method : {'gauss', 'curve'}, optional
        filter method used to smooth the input field

        * 'gauss' for 2d-Gaussian
        * 'curve' for curvature flow filter

    numberOfIterations : int, optional
        number of repeated application of curvature
        flow filter the larger then smoother if selected

    cluster_masking : {True, False}, optional
        if original (True) or smoothed (False) threshold mask is used.

    exclude_border : {True, False}, optional
        if clusters that touch domain borders are excluded (set to zero)

    marker_method : str, optional
        defines a method for marker assignment

        * 'brute_force_local_max'  : just uses scipy.ndimage.maximum_filter
        * 'skimage_peak_local_max' : uses skimage local maximum routine
        * 'mahotas_regmax'         : uses mahotas regional maximum detection
        * 'iterative_shrinking'    : uses iterative object shrinking depending on
          relative distance-to-background

    siggauss : float, optional
        sigma for Gaussian filter if selected

    marker_shrinkage_Niter : int, optional
        option for marker creation, how often shrinking is applied

    marker_shrinkage_dmin : float, optional
        option for marker creation

        How much of the edge is taken away. Distance is measured
        relative to the maximum distance, thus 0 < dmin < 1.


    Returns
    -------
    c : np.array
        categorial cluster field
    """

    # apply filtering ------------------------------------------------
    ndim = np.ndim(f)
    if filter_method == "curve":
        f_sm = curve_flow_filter(f, numberOfIterations=numberOfIterations)
    elif filter_method == "gauss":
        f_sm = scipy.ndimage.gaussian_filter(f, siggauss)
    # ================================================================

    # do masking -----------------------------------------------------
    ma = f > thresh
    ma_sm = f_sm > thresh
    # ================================================================

    # apply distance transform ---------------------------------------

    if marker_field == "dist":
        mfield = scipy.ndimage.distance_transform_edt(ma_sm)
    else:
        mfield = np.where(f_sm < thresh, 0, f_sm - thresh)
    # ================================================================

    # find local maximaand set markers -------------------------------
    if type(marker_field) == type(np.array([])):

        markers = marker_field
        print("...take predefined marker field")

    elif marker_method == "brute_force_local_max":

        mfield_max = scipy.ndimage.maximum_filter(mfield, 2 * dradius + 1)
        local_maxi = (mfield_max == mfield) & ma_sm
        markers, nclust = scipy.ndimage.label(local_maxi)

    elif marker_method == "skimage_peak_local_max":

        local_maxi = skimage.feature.peak_local_max(
            mfield, indices=False, exclude_border=exclude_border, min_distance=dradius
        )
        markers, nclust = scipy.ndimage.label(local_maxi)

    elif marker_method == "mahotas_regmax":

        local_maxi = mh.regmax(mfield, Bc=2 * dradius + 1)
        markers, nclust = scipy.ndimage.label(local_maxi)

    elif marker_method == "iterative_shrinking":

        # all segmentation / filter keywords are needed for marker determination
        marker_kwargs = kwargs.copy()

        marker_kwargs["ctype"] = ctype
        marker_kwargs["filter_method"] = filter_method
        marker_kwargs["numberOfIterations"] = numberOfIterations
        marker_kwargs["cluster_masking"] = cluster_masking
        marker_kwargs["siggauss"] = siggauss

        markers = markers_from_iterative_shrinking(
            ma_sm,
            marker_shrinkage_Niter=marker_shrinkage_Niter,
            marker_shrinkage_dmin=marker_shrinkage_dmin,
            **marker_kwargs,
        )

    # ================================================================

    # watersheding ----------------------------------------------------
    connect = set_connectivity_footprint(ctype, ndim)

    c = watershed(-mfield, markers, mask=ma_sm, connectivity=connect)
    # ================================================================

    # use original mask ----------------------------------------------
    if cluster_masking:
        c = np.where(ma, c, 0)
    # ================================================================

    return c


######################################################################
######################################################################


def connectivity_clustering(
    f,
    thresh,
    filter_method="curve",
    numberOfIterations=5,
    ctype=4,
    cluster_masking=True,
    siggauss=1.0,
    **kwargs,
):
    """
    Applies connectivity clustering to a field.


    Parameters
    ----------
    f : np.array
        field to be segmented

    thresh : float
        threshold for masking, set foreground to f > thresh

    filter_method : {'gauss', 'curve'}, optional
        filter method used to smooth the input field

        * 'gauss' for 2d-Gaussian
        * 'curve' for curvature flow filter

    numberOfIterations : int, optional
         number of repeated application of curvature
         flow filter the larger then smoother if selected

    ctype: {4, 8}, optional
        set either 4- or 8-connectivity (edge vs. edges+corners)

    cluster_masking : {True, False}, optional
         if original (True) or smoothed (False) threshold mask is used.

    siggauss : float, optional
         sigma for Gaussian filter if selected

    **kwargs : dict, optional
         set of additional, but not used keywords


    Returns
    -------
    c : np.array
        categorial cluster field

    """

    # apply filtering ------------------------------------------------
    ndim = np.ndim(f)
    if filter_method == "curve":
        f_sm = curve_flow_filter(f, numberOfIterations=numberOfIterations)
    elif filter_method == "gauss":
        f_sm = scipy.ndimage.gaussian_filter(f, siggauss)
    # ================================================================

    # do masking -----------------------------------------------------
    ma = f > thresh
    ma_sm = f_sm > thresh
    # ================================================================

    # connectivity ---------------------------------------------------
    connect = set_connectivity_footprint(ctype, ndim)
    c, nclust = scipy.ndimage.label(ma_sm, connect)
    # ================================================================

    # use original mask ----------------------------------------------
    if cluster_masking:
        c = np.where(ma, c, 0)
    # ================================================================

    return c


######################################################################
######################################################################


def watershed_merge_clustering(f, thresh, min_size=20, merge_ratio=0.5, **kwargs):
    """

    Applies watershed clustering to a field and then tries to merge them again.

    The merging is based on the ratio between maximum distances-to-background
    within the subclusters and on the cluster broders.


    Parameters
    ----------
    f : np.array
        field to be segmented

    thresh : float
        threshold for masking, set foreground to f > thresh

    min_size : int, optional, default = 20
        optional, minimum cluster size (px) left in the cluster field

    merge_ratio : float, optional
        essentially the ratio between interface length and cluster size

        if the interface is longer than `merge_ratio` * cluster_size
        than to two connected objects are merged.

    **kwargs : dict, optional
        keywords passed to `watershed_clustering` routine


    Returns
    -------
    c_update : np.array
        categorial cluster field (updated for merges)

    """

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Section 1: split the existing cluster into smaller parts
    # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    # split the field using watershed segmentation -------------------
    cw = watershed_clustering(f, thresh, **kwargs)

    # sort clusters ... THIS MIGHT BE REMOVED ....
    # cw = remove_small_clusters(cw, 0)
    # ================================================================

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Section 2:  get information about the borders
    # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    # use dilattion to expand the original field ---------------------
    cadd = scipy.ndimage.maximum_filter(cw, 3)

    # get edge or border from difference between the fields
    cedge = np.where(cw == 0, 0, cw - cadd)

    # the number of two touching clusters are extracted
    c1 = cw[cedge != 0]
    c2 = cadd[cedge != 0]

    # now we collect all possible combinations of touching clusters
    touching_clusters = set(zip(c1, c2))
    # ================================================================

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Section 3: test for merging
    # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    # get distance field ---------------------------------------------
    dist = scipy.ndimage.distance_transform_edt(f > thresh)
    # ================================================================

    # loop over all combinations -------------------------------------
    merging_clusters = []
    for nc1, nc2 in sorted(touching_clusters):

        # set cluster masks
        mask1 = cw == nc1
        mask2 = cadd == nc2

        # generate edge mask
        edge_mask = (cedge != 0) & mask1 & mask2

        # maximal distances-to-background within the clusters
        d1_max = dist[mask1].max()
        d2_max = dist[mask2].max()

        # now take the smaller one
        d_max = np.minimum(d1_max, d2_max)

        d_edge_max = dist[edge_mask].max()

        # calculate distance ratio
        dist_ratio = d_edge_max / d_max
        if dist_ratio > merge_ratio:
            merging_clusters += [(nc1, nc2)]
    # ================================================================

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Section 4: apply all merges
    # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    # get lists of connected clusters
    connected_clusters = connected_sequences2pairlists(merging_clusters)

    for nmerge in list(connected_clusters.keys()):

        # list of clusters that should be merged
        mlist = connected_clusters[nmerge]
        nc_target = mlist[0]
        for m in mlist[1:]:
            cw = np.where(cw == m, nc_target, cw)

    # this sorts the stuff agian
    c_update = remove_small_clusters(cw, min_size=min_size)

    return c_update


######################################################################
######################################################################


def connected_sequences2pairlists(connected_pairs_list):
    """
    The routine gets a set of paired numbers which show connection between numbers
    and collects all connections in a list of pairs.


    Parameters
    ----------
    connected_pairs_list : list
        list of number pairs


    Returns
    -------
    pair_list : dict
        dictionary that contains pair lists

    """

    merging_clusters = connected_pairs_list[:]

    pair_list = {}
    cluster_list_copy = merging_clusters[:]
    i = 1

    while len(cluster_list_copy) != 0:
        NOT_CONVERGED = True
        pair_list[i] = list(merging_clusters.pop(0))

        while NOT_CONVERGED:
            for pair in merging_clusters:
                c1, c2 = pair
                FOUND = False

                if c1 in pair_list[i]:
                    pair_list[i] += [
                        c2,
                    ]
                    FOUND = True

                if c2 in pair_list[i]:
                    pair_list[i] += [
                        c1,
                    ]
                    FOUND = True

                if FOUND:
                    cluster_list_copy.remove(pair)

            # check if something changed
            NOT_CONVERGED = merging_clusters != cluster_list_copy

            merging_clusters = cluster_list_copy[:]

        i += 1

    for npair in pair_list:
        pair_list[npair] = sorted(list(set(pair_list[npair])))

    return pair_list


######################################################################
######################################################################


def markers_from_iterative_shrinking(
    mask, marker_shrinkage_Niter=3, marker_shrinkage_dmin=0.1, **kwargs
):
    """
    The routine tries to find core parts of objects via iterative shrinking.

    Each object is normalized by its maximum distance-to-background which
    makes the algorithm scale-invariant.


    Parameters
    ----------
    mask : np.array
        binary mask


    marker_shrinkage_Niter : int, optional
        option for marker creation, how often shrinking is applied

    marker_shrinkage_dmin : float, optional
        option for marker creation

        How much of the edge is taken away. Distance is measured
        relative to the maximum distance, thus 0 < dmin < 1.


    Returns
    -------
    markers : np.array
         markers field with increasing index per marker region
    """

    # precalculate the distance transform ----------------------------
    d = scipy.ndimage.distance_transform_edt(mask)
    # ================================================================

    # connectivity analysis for segmentation -------------------------
    c = connectivity_clustering(d, 0, **kwargs)
    # ================================================================

    for niter in range(marker_shrinkage_Niter):
        print(niter, ". Shrinking iteration")

        # get maximum distance-to-background per object ------------------
        dmax = scipy.ndimage.measurements.maximum(
            d, labels=c, index=list(range(c.max() + 1))
        )
        # ================================================================

        # get normed distance -> max. = 1 per object ---------------------
        dnorm = np.where(dmax[c] == 0, 0, d / dmax[c])
        # ================================================================

        # perform segmentation on normed distance ------------------------
        cnorm = connectivity_clustering(dnorm, marker_shrinkage_dmin, **kwargs)
        # ================================================================

        # get new initial values -------------------------------------
        c = cnorm
        d = scipy.ndimage.distance_transform_edt(c != 0)
        # ============================================================

    # cluster field as output ----------------------------------------
    markers = c
    # ================================================================

    return markers


######################################################################
# (2) Multi-Treshold Clustering
######################################################################


def multithreshold_clustering(
    f, thresh_min, thresh_max, nthresh=10, use_percentile_threshold=False, **kws
):
    """
    Performs sequential segmentation and returns the combine result only keeping the
    predecessor object alive.


    Parameters
    ----------
    f : numpy array, 2dim or 3dim
        input field

    thresh_min : float
        minimum threshold value

    thresh_max : float
        maximum threshold value

    nthresh : int, optional
        number of threshold values

    use_percentile_threshold : {True, False}, optional
        if threshold list is derived from field percentiles

    **kws : dict
        keywords passed to segmentation routine


    Returns
    -------
    c : numpy array, same shape as f
        combined label field
    """

    # get the segmented data stack
    c3d = sequential_segmentation(
        f,
        thresh_min,
        thresh_max,
        nthresh=nthresh,
        use_percentile_threshold=use_percentile_threshold,
        **kws,
    )

    # do the sequential combination
    c = combine_object_stack(c3d)

    return c


######################################################################
######################################################################


def sequential_segmentation(
    f, thresh_min, thresh_max, nthresh=10, use_percentile_threshold=False, **kws
):
    """
    Performs sequential segmentation.


    Parameters
    ----------
    f : numpy array, 2dim or 3dim
        input field

    thresh_min : float
        minimum threshold value

    thresh_max : float
        maximum threshold value

    nthresh : int
        number of threshold values

    use_percentile_threshold : {True, False}, optional
        if threshold list is derived from field percentiles

    **kws : dict
        keywords passed to segmentation routine


    Returns
    -------
    c3d : numpy array, one dimension added
        stack of labeled field, from max. thresh to min. thresh
    """

    # get shape of field
    s = list(f.shape)

    # get threshold list
    if use_percentile_threshold:
        perc_list = np.linspace(thresh_min, thresh_max, nthresh)

        thresh_list = np.percentile(f, perc_list)
    else:
        thresh_list = np.linspace(thresh_min, thresh_max, nthresh)

    # inverse list from max to min
    thresh_list = thresh_list[::-1]

    shape3d = [
        nthresh,
    ] + s
    c3d = np.zeros(shape3d)
    noffset = 0

    # now loop over thresholds
    for i in range(nthresh):

        # get labels
        thresh = thresh_list[i]
        c = clustering(f, thresh, **kws)

        # store with offset
        c3d[i] = np.where(c == 0, 0, c + noffset)
        noffset = c3d[i].max()

    return c3d


######################################################################
######################################################################


def combine_object_stack(c3d):
    """
    Sequentially combines object label fields by searching for new objects
    that have no predecessor in the higher hierarchy level.


    Parameters
    ----------
    c3d : numpy array, 3dim or 4dim
        labeled object data stack


    Returns
    -------
    c : numpy array, 2dim or 3dim
        combined label field
    """

    # get shape and number of stacks
    s = c3d.shape
    nstack = s[0]

    c = c3d[0, :]

    for i in range(1, nstack):

        # extract two labeling levels
        c_prev = c  # 3d[i - 1].astype( np.int )
        c_next = c3d[i].astype(np.int64)

        # get indication if cell is already define for predecessor
        maxlab_prev = scipy.ndimage.measurements.maximum(
            c_prev, labels=c_next, index=np.arange(0, c_next.max() + 1)
        )

        # predecessor mask where a cell already exists
        pred_mask = maxlab_prev[c_next] > 0

        # and also get the background
        next_bg_mask = c_next == 0

        # combine the two possibilities
        mask = np.logical_or(pred_mask, next_bg_mask)

        c = np.where(mask, c, c_next)

    c = sort_clusters(c.astype(np.int64))

    return c


######################################################################
# (3) Utility Functions
######################################################################


def set_connectivity_footprint(ctype, ndim):
    """
    Return connectivity footprint for either 4-connectivity or
    8-connectivity.

    4-connectivity only includes neigboring sides, 8-connectivity also
    includes neighboring diagonal values.



    Parameters
    ----------
    ctype: {4, 8}
        set either 4- or 8-connectivity (edge vs. edges+corners)

    ndim : {2,3}
        dimension of the field (either 2-dim or 3-dim)


    Returns
    -------
    footprint : np.array
        ndim-dimensional connectivity footprint
    """

    if ctype == 4:

        if ndim == 2:
            footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        elif ndim == 3:
            footprint = np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )

    elif ctype == 8:
        footprint = np.ones(
            [
                3,
            ]
            * ndim
        )

    return footprint


######################################################################
######################################################################


def sort_clusters(c):
    """
    It sorts clusters by size (number of pixels) in decreasing order.


    Parameters
    ----------
    c : np.array
        categorial cluster field


    Returns
    -------
    c_re : np.array
        categorial cluster field with sorted objects (from large to small)
    """

    c_re = remove_clustersize_outside(c, min_size=0, max_size=np.inf)
    return c_re


######################################################################
######################################################################


def remove_small_clusters(c, min_size=6):
    """
    Counts numbers of pixels per cluster cell and removes clusters smaller
    than minimum size.

    In addition, it sorts clusters by size in decreasing order.


    Parameters
    ----------
    c : np.array
        categorial cluster field

    min_size : int, optional
        minimum cluster size (px) left in the cluster field


    Returns
    -------
    c_re : np.array
        categorial cluster field with sorted objects (from large to small)

        objects smaller than or equal `min_size` are removed
    """

    return remove_clustersize_outside(c, min_size=min_size)


######################################################################
######################################################################


def remove_clustersize_outside(c, min_size=6, max_size=np.inf):
    """
    Counts numbers of pixels per cluster cell and removes clusters outside
    a size range defined by a minimum and maximum size.

    In addition, it sorts clusters by size in decreasing order.


    Parameters
    ----------
    c : np.array
        categorial cluster field

    min_size : int, optional
        minimum cluster size (px) left in the cluster field

    max_size : int, optional
        maximum cluster size (px) left in the cluster field


    Returns
    -------
    c_re : np.array
        categorial cluster field with sorted objects (from large to small)

        * objects smaller than or equal `min_size` are removed
        * objects larger than or equal `max_size` are removed

    """

    # set binary mask
    mask = np.where(c == 0, 0, 1)

    # get pixel sums per object
    psum = scipy.ndimage.measurements.sum(
        mask, labels=c, index=list(range(c.max() + 1))
    )

    # how is pixel sum sorted? Biggest first ...
    isort = psum.argsort()[::-1]

    # set up the scoring of pixel sums
    score = np.arange(len(psum)) + 1

    # get the mapping
    imap = np.zeros_like(psum).astype(np.int64)
    imap[isort] = score

    # conditions
    imap[psum <= min_size] = 0
    imap[psum >= max_size] = 0

    c_re = imap[c]

    return c_re


######################################################################
######################################################################


def remove_clustersize_outside_slow(c, min_size=6, max_size=np.inf, sort=True):
    """
    OLD VERSION --> much slower ...

    Counts numbers of pixels per cluster cell and removes clusters outside
    a size range defined by a minimum and maximum size.

    In addition, it sorts clusters by size in decreasing order.


    Parameters
    ----------
    c : np.array
        categorial cluster field

    min_size : int, optional
        minimum cluster size (px) left in the cluster field

    max_size : int, optional
        maximum cluster size (px) left in the cluster field

    sort : {True, False}, optional
       if objects are sorted by size


    Returns
    -------
    c_re : np.array
        categorial cluster field with sorted objects (from large to small)

        * objects smaller than or equal `min_size` are removed
        * objects larger than or equal `max_size` are removed
    """

    # calculate cluster sizes ----------------------------------------
    clust_size = {}

    # init new number index and cluster field
    c_re = np.zeros_like(c)
    cnum_new = 1

    # also init mapping between size and number of field
    csizes = []

    for cnum in range(1, c.max() + 1):

        # set mask
        m = c == cnum

        # get size
        csize = len(c[m])
        clust_size[cnum] = csize

        # assign new number to cluster that is large enough
        if csize >= min_size and csize <= max_size:

            c_re[m] = cnum_new
            csizes.append(csize)

            cnum_new += 1

    # sort the field by cluster size
    if sort:

        csizes = np.array(csizes)
        cindices = csizes.argsort()[::-1]

        csort = np.zeros_like(c)

        for ci_new, ci_old in enumerate(cindices):
            m = c_re == ci_old + 1
            csort[m] = ci_new + 1

        cout = csort
    else:
        cout = c_re
    # ================================================================

    return cout


######################################################################
######################################################################


def percentiles_from_cluster(f, c, p=[25, 50, 75], index=None):
    """
    Calculates percentiles of cells in an segmented (labeled) field.

    Functionality missing in scipy.ndimage.measurements.

    Parameters
    ----------
    f : np.array
        the field as basis for percentile calculation

    c : np.array
        categorial cluster field

    p : list or np.array, optional
        percentiles array

    index : list or `None`, optional
        list of object labels for which percentiles are calculated


    Returns
    -------
    pc : np.array
        percentiles per cell (including background (set to zero))
    """

    # get slices .....................................................
    slices = scipy.ndimage.measurements.find_objects(c)

    # set index
    if index == None:
        index = list(range(1, len(slices) + 1))

    # loop over slices ...............................................

    nperc = len(p)
    pc = [
        np.zeros(nperc),
    ]

    for i in index:

        try:
            s = slices[i - 1]
            cc = c[s]
            fc = f[s]

            m = cc == i  # assume that slices are sorted

            pvec = np.percentile(fc[m], p)
        except:
            pvec = np.nan * np.ones(nperc)

        pc.append(pvec)

    return pc


######################################################################
######################################################################


if __name__ == "__main__":

    # do the test with random field ----------------------------------
    r = np.random.randn(100, 100)
    r_sm = scipy.ndimage.median_filter(r, 5)
    rm = np.where(r_sm > 0.1, r_sm, 0)
    # ================================================================

    # connectivity ---------------------------------------------------
    c_con = connectivity_clustering(rm, 0)
    c_wat_mark = watershed_clustering(rm, 0)
    c_wat = watershed_clustering(rm, 0, marker_field="same", dradius=6)
    # ================================================================

    import matplotlib

    matplotlib.use("Agg")
    import pylab as pl

    pl.subplot(221)
    pl.imshow(np.ma.masked_equal(rm, 0))
    pl.subplot(222)
    pl.imshow(np.ma.masked_equal(c_con, 0), cmap=pl.cm.prism)
    pl.subplot(223)
    pl.imshow(np.ma.masked_equal(c_wat_mark, 0), cmap=pl.cm.prism)

    pl.subplot(224)
    pl.imshow(np.ma.masked_equal(c_wat, 0), cmap=pl.cm.prism)
