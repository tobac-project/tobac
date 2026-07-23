import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tobac.eroded_contiguity as tb_ec


def test_apply_periodic_boundary():
    """test behaviour of apply_periodic_boundary is as expected for the three PBC options. """
    x = np.array([
        [1,0,0,2],
        [1,0,0,2],
        [0,0,0,0],
        [3,0,0,0],], dtype=int)
    
    expected_hdim_1 = np.array([
        [1,0,0,2],
        [1,0,0,2],
        [0,0,0,0],
        [1,0,0,0],], dtype=int)
    
    out_hdim_1 = tb_ec.apply_periodic_boundary(x.copy(), 'hdim_1')
    assert (out_hdim_1 == expected_hdim_1).all()
    
    expected_hdim_2 = np.array([
        [1,0,0,1],
        [1,0,0,1],
        [0,0,0,0],
        [3,0,0,0],], dtype=int)
    
    out_hdim_2 = tb_ec.apply_periodic_boundary(x.copy(), 'hdim_2')
    assert (out_hdim_2 == expected_hdim_2).all()

    expected_both = np.array([
        [1,0,0,1],
        [1,0,0,1],
        [0,0,0,0],
        [1,0,0,0],], dtype=int)
    
    out_both = tb_ec.apply_periodic_boundary(x.copy(), 'both')
    assert (out_both == expected_both).all()


def test_calculate_object_topography():
    """Test the method works on a simple blob. """
    arr = np.zeros((7, 7), dtype=int)
    arr[1:6, 1:6] = 1
    
    topo = tb_ec.calculate_object_topography(1, arr)
    
    assert topo.shape == arr.shape
    assert topo.max() == pytest.approx(1.0)
    assert topo[3,3] == pytest.approx(1.0)
    assert topo[1,1] < topo[3, 3]
    assert topo[0,0] == pytest.approx(0.0)


def simple_mask_4d():
    data = np.zeros((2, 2, 5, 5), dtype=int)
    data[0, 0, 1:4, 1:4] = 1
    data[1, 0, 1:4, 1:4] = 1
    data[0, 1, 1:4, 1:4] = 1
    data[1, 1, 1:4, 1:4] = 1
    return xr.DataArray(
        data,
        dims=('time', 'z', 'y', 'x'),
        coords={
            'time': [0,1],
            'z': [0, 1],
            'y': [0, 1, 2, 3, 4],
            'x': [0, 1, 2, 3, 4],
        },
        name='mask',
    )


def less_simple_mask_4d():
    data = np.zeros((2, 2, 5, 5), dtype=int)
    data[0, 0, 0:3, 0:3] = 1
    data[0, 0, 4:5, 0:3] = 1
    data[0, 1, 0:3, 0:3] = 1
    data[0, 1, 4:5, 0:3] = 1
    data[1, 0, 1:4, 1:4] = 1
    data[1, 1, 1:4, 1:4] = 1
    return xr.DataArray(
        data,
        dims=('time', 'z', 'y', 'x'),
        coords={
            'time': [0,1],
            'z': [0, 1],
            'y': [0, 1, 2, 3, 4],
            'x': [0, 1, 2, 3, 4],
        },
        name='mask',
    )


def test_calculate_mask_topography():
    """Test the method works on a simple blob but with vertical and time dimensions. """
    mask = simple_mask_4d()
    
    topo = tb_ec.calculate_mask_topography(mask)
    
    assert topo.shape == mask.shape
    assert topo.max() == pytest.approx(1.0)
    assert topo[0, 0, 2, 2] == pytest.approx(1.0)
    assert topo[0, 0, 1, 1] < topo[0, 0, 2, 2]
    assert topo[0, 0, 0, 0] == pytest.approx(0.0)
    assert (topo[1] == topo[0]).all()


def test_erode_mask():
    """Test the method works and test PBC conditions. """
    mask = simple_mask_4d()
    eroded = tb_ec.erode_mask(mask, fraction=.75, vdim="z", use_parallel=0)
    
    assert eroded.values[0, 0, 0, 0] == 0
    assert eroded.values[0, 0, 2, 2] == 1
    assert eroded.values[0, 0, 1, 1] == 0
    assert (eroded[1] == eroded[0]).all()

    mask = less_simple_mask_4d()
    eroded = tb_ec.erode_mask(mask, fraction=.75, vdim="z", PBC_flag=None)
    
    assert (eroded.where(mask==0, 0) == 0).all()
    assert eroded.values[0, 0, 1, 1] == 1

    mask = less_simple_mask_4d()
    eroded = tb_ec.erode_mask(mask, fraction=.75, vdim="z", PBC_flag="both")
    
    assert (eroded.where(mask==0, 0) == 0).all()
    assert (eroded.values[0, 0, 0:2, 1] == 1).all()

    mask = less_simple_mask_4d()
    eroded = tb_ec.erode_mask(mask, fraction=.75, vdim="z", PBC_flag="hdim_1")
    
    assert (eroded.where(mask==0, 0) == 0).all()
    assert (eroded.values[0, 0, 0:2, 1] == 1).all()

    mask = less_simple_mask_4d()
    eroded = tb_ec.erode_mask(mask, fraction=.75, vdim="z", PBC_flag="hdim_2")
    
    assert (eroded.where(mask==0, 0) == 0).all()
    assert eroded.values[0, 0, 1, 1] == 1


def multiple_blobs_4d():
    data = np.zeros((2, 1, 10, 10), dtype=int)
    data[0, 0, 0:3, 0:3] = 1
    data[0, 0, 8:, 0:3] = 2
    data[0, 0, 5:6, 6:9] = 3
    data[1, 0, 1:4, 1:4] = 4
    data[1, 0, 5:6, 6:9] = 5
    return xr.DataArray(
        data,
        dims=('time', 'z', 'y', 'x'),
        coords={
            'time': [0,1],
            'z': [0],
            'y': np.arange(0,10,1),
            'x': np.arange(0,10,1),
        },
        name='cell',
    ), pd.DataFrame({"cell":[1,2,3,4,5]})


def test_track_using_contiguity():
    """Test the method works and test PBC conditions. """
    mask = simple_mask_4d()    
    tracked = tb_ec.track_using_contiguity(mask, vdim="z", use_parallel=0)
    
    assert tracked.shape == mask.shape
    assert tracked.values[0, 0, 2, 2] == 1
    assert tracked.values[0, 0, 1, 1] == 0
    assert (tracked[1] == tracked[0]).all()

    mask, table = multiple_blobs_4d()
   
    tracked_mask, tracked_table = tb_ec.track_using_contiguity(mask, table, vdim="z", PBC_flag="both")
    expected_tracks = pd.DataFrame({"cell":[1,2,3,4,5], "contiguous":[1,1,2,1,2]})

    assert (tracked_table == expected_tracks).all().all()

    tracked_mask, tracked_table = tb_ec.track_using_contiguity(mask, table, vdim="z", PBC_flag=None)
    expected_tracks = pd.DataFrame({"cell":[1,2,3,4,5], "contiguous":[1,3,2,1,2]})

    assert (tracked_table == expected_tracks).all().all()


def test_track_using_eroded_contiguity():
    """Test the method works and test PBC conditions. """

    mask, table = multiple_blobs_4d()
    mask[0, 0, 8:, 0:3] = 2

    tracked_table = tb_ec.track_using_eroded_contiguity(mask, table, fraction=.25, vdim="z", PBC_flag="both",)
    expected_tracks = pd.DataFrame({"cell":[1,2,3,4,5], "contiguous":[1,1,2,1,2]})

    assert (tracked_table == expected_tracks).all().all()

    tracked_table = tb_ec.track_using_eroded_contiguity(mask, table, fraction=.75, vdim="z", PBC_flag="both",)
    expected_tracks = pd.DataFrame({"cell":[1,2,3,4,5], "contiguous":[1,3,2,4,2]})

    assert (tracked_table == expected_tracks).all().all()

    tracked_table = tb_ec.track_using_eroded_contiguity(mask, table, fraction=.25, vdim="z", PBC_flag=None,)
    expected_tracks = pd.DataFrame({"cell":[1,2,3,4,5], "contiguous":[1,3,2,1,2]})

    assert (tracked_table == expected_tracks).all().all()